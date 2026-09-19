# E038 - two depths pin the run shape, and the chain is majority prefill at depth

Same command as E037 with `-d 131072` instead of `-d 40960`, both arms (`stats-qsa-131k.log`,
`stats-no-qsa-131k.log`). `rtile` off, so still dense vec FA in both arms.

> **Trust ratios, not absolute ms:** the stats table's agent coverage and whether `--stats`
> replays kernels are unconfirmed, so absolute millisecond figures are unknown up to a constant.
> The cache-placement section below supplies the wall-clock bound that keeps the per-card
> reading honest (1.5x tg from dropping the chain rules out 4x replication).

## The run shape falls out of the call counts

`cpy_scalar_transpose` is built twice per prefill ubatch per QSA layer, `fill_kernel<__half>` once per
chain invocation per layer. Dividing by 12 layers:

| | @40960 | @131072 | x |
| --- | --- | --- | --- |
| prefill builds (transpose / 2 / 12) | 160 | 512 | 3.20 |
| all chain builds (fill / 12) | 1,700 | 2,052 | 1.21 |
| **decode builds** | **1,540** | **1,540** | **1.00** |

160 = 4 passes x 40960/1024 and 512 = 4 x 131072/1024: so `-d N -p 0` *does* prefill to depth in
`-ub 1024` ubatches, over 4 passes (3 reps + warmup), and the decode build count is depth-independent.
1,540 decode builds for 128 x 4 tokens = **3.0 chain builds per decoded token**, which answers the
4.4x question from E037 (3, not MTP). Cross-checked by four independent count ratios: `mm_ids_helper`
x3.20, the prefill Tensile GEMM x3.20, `mul_mat_vec_q<12,1>` x1.00, `gated_delta_net` x1.21.

## Cache placement under `-sm tensor`, settled by the load log

The user ran `llama-bench ... -sm tensor -d 8192 -p 0 -n 8 -r 1 -v` on the bench box and grepped the
cache lines. Main attention cache: `size = 198.00 MiB (8448 cells, 12 layers)` with
`Meta() KV buffer size = 49.50 MiB` = 198/4 -> **split four ways**. Indexer cache:
`creating indexer KV cache, size = 8448 cells`, `size = 24.75 MiB (8448 cells, 12 layers)`,
`K (f16): 24.75 MiB`, buffer line `24.75 MiB` = the whole thing -> **mirrored, a full copy per device**,
which is what `src/llama-model.cpp:511-514` declares for `cache_idx_(k|v)_l*` ("the qsa indexer has one
key head and its projections are mirrored, so its cache cannot be split") - a rule that arrived with
upstream's own `6c84c7d5d model: add Qwen3.8-Flash-Next (qwen4exp)`. So H17's correctness half is
closed: every device sees all cells and the top-k is global, which is also what the user's bench-side
probe line showed (`k=[256,4352,2] nkv=4352 gqa=12`). The mirror costs ~384 MiB/card at 131k, ~1.5 GiB
across the box.

~~The 4x-redundant-chain reading is ruled out by wall clock~~ **withdrawn the same day.** The argument
went: 12 layers x ~2.8 ms = ~33 ms/token of a 45 ms step would predict a 3-4x tg gain from dropping the
chain, and 1.5x was measured. But the 2.8 ms is *my dev box's* per-layer cost, and the argument silently
assumed a card there costs the same as a card here. If an R9700 is ~2x an RX 9070, a fully replicated
chain is ~17 ms/token, which also fits 15.5 ms. So replication is not excluded, and whether the chain's
~41 nodes per QSA layer are copied into all four device sub-graphs is open (H17b). What is *not* open:
the graph is partitioned per device, because the fork's own `a8b24dfdf` works around exactly that -
upstream had `case LLM_ARCH_QWEN4EXP: // TODO: fix test-llama-archs` on the `-sm tensor` denylist, that
commit deletes it and adds `ggml_build_forward_expand(gf, res_hc)` at `src/models/qwen4exp.cpp:390` to
force `hc_init` into the same graph split as layer 0. So the upstream refusal is a test failure, not an
architecture ruling, and the consequence for this record is that its per-card numbers are only valid
under one of two placement models.

## Chain cost, split by phase

Two equations (chain total at each depth), with prefill chain work quadratic in ubatch count (each
ubatch's cost is linear in the then-current `n_kv`, `n_tps` fixed at 1024) and decode work linear in
`n_kv`:

| | @40960 | @131072 |
| --- | --- | --- |
| chain, prefill part | 6.94 s (47%) | 71.0 s |
| chain, decode part | 7.84 s (53%) | 25.1 s |
| chain total | 14.78 s | 96.11 s (x6.5) |
| decode chain per build, 4 cards summed | 5.09 ms | 16.29 ms |
| same, per card | 1.27 ms | 4.07 ms = **0.34 ms per QSA layer** |

So the x6.5 growth of the chain while `n_kv` only grows x3.2 is **a mix effect, not a cliff**: the extra
prefill ubatches at 131k are the deep, expensive ones. The per-call table shows the same thing: the
cache-shaped terms grow as predicted (`gather f16->f32` x2.96, `rope_multi` x2.86, `mul_mat_vec_f<4,64>`
x2.87, `scale_f32` x2.74, `rms_norm` x2.74, `top_k_radix_select` x2.64, expand gather x2.42,
`cpy_scalar_transpose` x3.50) while the terms whose *prefill* shape carries an extra `n_tps` factor grow
x6-10 (`k_bin_bcast` f32 add x8.68, `relu` x8.56, f16 mask add x7.96, `top_k_radix_histogram` x7.46,
`copyBufferRect` x5.85, `fill_kernel<__half>` x7.10, `cpy_scalar_contiguous` x7.80). No RDNA-specific
behaviour needed to explain any of it.

Caveat on the fit: it assumes each chain term is cache-shaped. `top_k`'s 11 launches have a fixed part,
so the prefill/decode split is approximate. The control fit on the *whole* run is invalid and was
dropped - prefill time is dominated by weight GEMMs, which do not scale with `n_kv`.

## Decode: device time now accounts for the wall-clock effect

Per decoded token at 131k: 3 builds x 16.29 ms summed / 4 cards = **12.2 ms/token**, against the
15.5 ms/token gain measured from `Q4EXP_NO_INDEXER` (22.24 -> 33.89 t/s). So **79% of the gain is chain
device time** and the dispatch hypothesis (H14) is closed - it stays only as the residual 21%, which is
within the error of the phase split.

And 0.34 ms per QSA layer per card at 131k, against 2.78 ms per QSA layer per step for *one* card on the
dev box at 164k, is consistent with the chain being split ~4 ways by `-sm tensor` (expected ~0.7 ms,
measured ~0.34 ms with the R9700 being the faster card). See H17: that is a correctness question, not
just a cost one.

## NCCL at 131k

22.0 s -> 95.5 s with calls x1.21, so per call 136 -> 485 us (x3.6) even though a 2560-wide f32 allreduce
carries the same bytes at any depth: that is peer-wait time, i.e. the imbalance exposure grows with
depth. At 131k it is ~15 ms/token of the ~45 ms/token step, the same order as the whole chain. H16.

## Priorities as they now stand

- **H13 is a prefill item.** At 131k the chain is 74% prefill and the terms H13 shrinks or deletes are
  the prefill-shaped ones (`top_k` histogram over `n_kv x n_tps`, the cell expand, the permute pair, the
  f32 adds, `relu`). Ceiling: 71 s of 430 s total device time, so ~+16% pp at 131k if all of it went.
- **H9 is the decode item**, ~25 s at 131k, of which the gather (5.3 s), rope (4.6 s), norm (1.9 s) and
  the strided slices (6.6 s) are the parts that vanish if block keys are pooled once at write time.
- H15 (f16 staging) still stands: the f16 mask add and `cpy_scalar_contiguous<__half, float>` are both in
  the fast-growing family.
