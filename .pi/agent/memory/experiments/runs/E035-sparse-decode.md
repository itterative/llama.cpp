# E035 - sparse decode exists now, and it pays

Continuation of E034. `fattn-rtile` walks the QSA selection instead of the cache, so for the first
time in this branch decode can stop reading every KV row. Commit `d262224eb`, WIP, off unless
`GGML_FATTN_RDNA_RTILE=1`.

## What the kernel change is

`use_sparse` is a template parameter, so the dense arm compiles to what it did before. In the sparse
arm `KV_max_ptr` is re-read as an `int32` index row and `ne11` is `n_kv_max`, both of which
`launch_fattn` already arranges; the loop bound, the `k0*stride` term in the three staging calls, and
the mask read are the only uses of the old row index. A tile's rows come from `idx_row + k0`, and a
`-1` entry (the padding `flash_attn_mask_to_sparse_indices` writes past the count) or a row past the
end of the short last tile reads KV row 0 and gets `-INFINITY` from the mask, so the arithmetic
discards it without ever touching an out-of-range row. The tail bound is `min(nbatch_fa, k_VKQ_max - k0)`
because the index buffer is sized `n_kv_max * mask_rows` with no padding to the 32-row tile.

One property of QSA makes this cheap where it would not be for a per-head selection: the index row is
per (sequence, query token), and all 12 query heads of a KV head share it, so one gathered tile serves
every Q column the block owns.

## Validation, dev box, gfx1201, ROCm 7.1.1

Reference coverage, `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 -p 'nb=1,.*n_kv_max=[1-9]'`:
7 sparse decode cases, all OK with the arm on and off. Six are rtile-eligible by the shape gate - D
128 f16, D 128 q8_0, D 256 at two budgets, D 512 at two budgets - and all pass. This is the coverage
E032 said the sparse path had never had: a wrong row index cannot pass, because the CPU reference
honours the mask. The full FA suite still fails only in the known `hsk=192,hsv=128` family, 8 cases,
though the *error values* in that family move between two runs of different binaries, so treat that
family as nondeterministic and do not use its count as a gate.

`tg` on the shape-faithful dummy, both arms with `Q4EXP_SPARSE_FA=1`, so the only difference is
whether decode uses the selection:

| depth | dense decode | sparse decode | delta | KV bytes removed per token |
|---|---|---|---|---|
| 8192   | 235.51 +- 3.32 | 236.25 +- 3.33 | +0.3%  | 12.6 MB |
| 40960  | 199.20 +- 0.83 | 207.63 +- 1.01 | **+4.2%**  | 79.7 MB |
| 163840 | 125.99 +- 1.34 | 139.40 +- 0.66 | **+10.6%** | 332 MB |

The gain is depth-proportional and the implied marginal rate is consistent: 79.7 MB in 0.203 ms and
332 MB in 0.766 ms both come out near **420 GB/s** against a ~650 GB/s card. So the removed reads were
being paid at nearly streaming efficiency and the modest percentages are only because the dummy has one
attention layer in a four-layer model and a 5-8 ms step to begin with.

## What to expect on the bench, and how to tell if it failed to engage

Scaling the same byte saving to 12 attention layers at 131k removes ~3.19 GB per token out of the 45 ms
step. Carrying the marginal rate across gives roughly +20% to +30% on `tg` at 131k and +8% to +12% at
40k, nothing at 4k where the gate is shut. The rate is the weak link in that estimate: the bench's own
marginal rate, fitted from its depth curve, is nearer 200-290 GB/s than 420, which is what makes the
upper end possible and the lower end likely.

Two ways this silently measures zero:
- `-sm tensor` may leave a per-device `gqa_ratio` that is odd, and the gate requires even;
- `llama-bench` needs `-v` to show ggml logs, and needs `Q4EXP_SPARSE_FA=1` as well as the kernel flag.

`tools/rtile-fa-probe.patch` prints the shapes of the ops that took the sparse arm, once per kernel
instantiation. Run it before spending time on the perf number.

## Addendum - the bench saw nothing, and the gate is the first suspect

`results/user/results-sparse-decode.log`, build `d77fb8d53` (11083), both flags, `-sm tensor`, tg only:

| depth | this run | E027c3 sparse+vec | E027c3 dense |
|---|---|---|---|
| 4096   | 34.79 +- 1.73 | 35.09 | 36.13 |
| 16384  | 33.23 +- 1.64 | 33.16 | 34.00 |
| 40960  | 30.20 +- 1.23 | 30.17 | 30.97 |
| 131072 | 22.24 +- 0.69 | 21.63 | 21.99 |

So rtile moved nothing except 131k by +2.8%, which is inside its own error bar. Landed exactly on the
sparse+vec arm, i.e. the same number you get if the kernel is never selected.

The dev box proves the mechanism works: the gate probe prints
`q=[256,1,24] k=[256,4352,2] gqa=12 nkv=4352 n_kv_max=2051 -> TAKE` and tg pays +4.2%/+10.6% at
40k/164k. What is unknown is the **per-device** `gqa` under `-sm tensor`, and rtile can only express
even ratios: the dispatch splits q heads into `ncols2` in {16,8,4,2} groups sharing one staged KV row,
so a ratio of 3 would silently mis-group them. 24 q heads and 2 KV heads over 4 cards is either 6 q
per card with 1 KV head (ratio 6, fine) or 6 q with both KV heads replicated (ratio 3, skipped).

Two ways to settle it, in ascending cost:
- run the same matrix with `-sm layer`, where each card holds whole layers and the ratio stays 12. If
  the win appears there, tensor split's per-layer collectives were hiding it *and* the gate was the
  blocker on the tensor arm;
- `tools/rtile-gate-probe.patch`, which prints the shapes and TAKE/skip once per process regardless of
  the outcome. Needs `-v`, since llama-bench swallows ggml logs.

`tools/rtile-fa-probe.patch` (the one that prints only when the sparse arm is taken) is superseded by
the gate probe for this question.

## Still open

- whether to keep the kernel at all: dense rtile is a wash versus vec on RDNA4, so its stated purpose -
  consuming q8_0 and q4_0 KV natively - is the only thing it uniquely offers, and sparse decode is now
  a second one;
- the equivalence test that would close the numerics story end to end: patch the dummy's
  `indexer_budget` in place to exceed any cache length, so the selection is "everything visible" and
  sparse decode must reproduce dense decode exactly;
- H9, which removes the other context-proportional term in decode (the indexer re-pool, ~0.8-1.0 GB per
  token at 131k) and is untouched by any of this.
