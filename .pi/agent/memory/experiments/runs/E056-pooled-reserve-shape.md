# E056 - reserving the pooled qsa shape removes the prefill re-reservation churn

- date: 2026-09-25
- machine: dev-rx9070-16g (T1) + bench-4x-r9700-32g (T2, run by the user)
- tier: T1+T2
- status: done
- parent: E049 (mechanism), E051 (d1 landed, d2 tried and reverted), E052 (width policy)
- commit: `b811339d6` (the code); the raw output below is deliberately **not committed** - it is in a
  stash, see `raw`
- build: `-O2 -g3 -fno-omit-frame-pointer`, asserts on, no `-DNDEBUG`
- model: `models/q4exp-48l-12qsa.gguf` (perf), `models/q4exp-4l.gguf` (gates)

## hypothesis

The pp regression from pooling prefill is entirely the reservation mismatch: `sched_reserve` builds
through a full-cache context whose indexer cells are empty, so `qsa_pool_get` answers NONE and the
measure pass sizes the *historic* graph. If the reservation measures the pooled worst case instead,
prefill keeps the pool and stops re-reserving.

## prediction

Written before the A/B/C, after the probe below had identified the mechanism.

- `sched:realloc_*` per llama-bench invocation: 24+ (E052 arm G) -> **0**
- pp8192 @ d16384: from -2.9% vs pool-off (E052 G vs F) to **within noise**
- tg128: unchanged, still ~+4% over pool-off
- reserved compute buffer: within ~1 MiB of the historic reservation
- falsified if any `sched:realloc_size` survives in a pp table, or pp stays > 1% under pool-off

## what the probe showed (the part E051 did not have)

Two temporary ggml probes, both reverted before any number below was measured (`git checkout
ggml/src/ggml-alloc.c ggml/src/ggml-backend.cpp`): a dump of `sched->graph` per split, and a
`ggml_gallocr_needs_realloc` that reports *every* mismatch instead of returning at the first.

`q4exp-24l-6qsa`, `-sm tensor -p 2048 -d 4096 -ub 1024 -r 1 --no-warmup`, `Q4EXP_POOLED=1`:

- the reservation measures **3457 nodes / 692 leafs**, every runtime graph is **3483 / 700**. So the
  pooled and historic topologies differ by 26 nodes and 8 leafs on a 6-QSA-layer model, and the
  pooled shape is *stable*: all six prefill ubatches and both decode graphs dump identical counts.
- the first ubatch trips `backend_ids_changed` (topology), and every ubatch after it trips the size
  test on `leaf_118` - which is `inp->bias`, f32 `[n_blocks, n_tps]`, the block bias - plus
  `attn_inp_kq_mask` and the `Meta(ROCm0)#...#0` copies of both. Recorded `size_max` always equals
  the *previous* ubatch's size: that is E049's ratchet, and it is why one mismatch costs the whole
  pass rather than one ubatch.
- cost on this box: `graph:alloc` 6 calls / 2224 ms (5 `realloc_size` + 1 `realloc_buft`) against
  `graph:compute` 6 calls / 990 ms.
- after the fix, the runtime split-graph dumps are **byte-identical** to the pre-fix ones
  (`diff` on all 8 runtime dumps: no differences). The change touches only what the reservation
  measures, which is why the numerics cannot move.

## the second hole, found by the fix itself

d2 alone left 17 re-reserves in a `llama-perplexity -c 8192` run. Cause: `common_init`'s warmup
decodes 1-2 tokens on an empty cache, so `n_bid == 0`, and `qsa_pool_get` returned NONE for that
("nothing complete to pool yet"). One historic graph is enough to drop the pooled reservation, and
the ratchet does the rest. Same hole in llama-bench's `tg` warmup (`test_gen(ctx, 1)`).

So the short-run case now pools too: `n_new = max(1, n_bid)`, and `try_contiguous` derives row 0 from
the head of the run (cells clamped to `j1 - 1`, position `p0`). This cannot change any output, and
not only because the gates say so: while `n_bid == 0` every entry of the block bias is `-INFINITY`
(`set_input_qsa`: `b < n_bid_last && ... ? 0.0f : -INFINITY`), so `score = relu(dot) + (-inf)` is
`-inf` for every block whatever the row holds, and `top_k` sees the same input in both variants. The
row only has to be finite, which it is: its members are cells this ubatch just wrote.

## results - mechanism and perf, dev box, 3 interleaved passes x `-r 2`

All arms: `Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1` (the run8 config), `-ngl 99 -sm tensor -fa 1
-lzm on-direct -d 16384 -p 8192 -n 128 -ub 1024 -b 2048`, `GGML_PROF_REGIONS=1`, same session.

| arm | config | pp8192 median (min-max) | tg128 median (min-max) | re-reserves / invocation | reserve |
| --- | --- | --- | --- | --- | --- |
| A | `Q4EXP_POOLED=0` | 702.08 (701.52-704.40) | 34.38 (34.03-34.85) | 0 | 6775/1358, 1110.38 MiB |
| B | `+Q4EXP_POOLED_NO_PREFILL=1` | 702.14 (701.38-704.81) | 35.81 (35.77-36.07) | 2 (25.4 + 24.6 ms) | 6775/1358, 1110.38 MiB |
| C | pool at all widths (**new**) | 702.43 (701.86-703.52) | 35.86 (35.78-36.28) | **0** | 6825/1372, 1110.46 MiB |

- C vs A: pp **+0.05%**, tg **+4.3%**. C vs B: pp +0.04%, tg +0.1%, two fewer re-reserves.
- The reservation costs **+0.08 MiB**. The pooled worst case derives `n_new = n_bid` rows, which is
  the same work the historic chain does for `n_blocks` rows, so the buffer barely moves.
- Arm B's two re-reserves are new (E052 arm I had one): its `tg` warmup now pools the 1-token
  empty-cache decode, which no longer matches B's historic reservation. Arm C has none.
- Same-session spread is <= 0.5% on pp and <= 2.4% on tg, so pp is a wash by measurement and tg's
  +4% is the pool's known decode win. **Pooled prefill is device-time-neutral here**, which repeats
  E051's finding - but now measured without the churn hiding it.
- Also run at `-p 2048 -d 4096` on `q4exp-24l-6qsa` and with `Q4EXP_SPARSE_FA=0`: 0 re-reserves in
  every arm-C variant, so the result is not specific to one shape or to sparse FA.

## results - correctness gates (all with the final clean build)

| gate | pool off | pool on |
| --- | --- | --- |
| golden corpus, `-sm none` | 263113.6984 | 263113.6984 |
| sparse corpus depth sweep `-sm none -b 2048`, `-c` 2048/3072/4096/6144/8192 | 266436.6135 / 267361.5611 / 264605.3438 / 269600.9861 / 267035.3875 | identical, all five |
| sparse corpus `-sm tensor -c 8192`, dense FA | 267035.1801 | 267035.1801 |
| sparse corpus `-sm tensor -c 8192`, `Q4EXP_SPARSE_FA=1` | 267035.1320 | 267035.1320 |
| cached-heavy `-b 256 -c 2048` (121 pooled `set_input_qsa` calls) | 266571.9557 | 266571.9557 |
| 2500-token greedy decode, `-c 4096 -s 7` (2504 pooled calls) | 43798 B of text | byte-identical |
| context-shift stress `-c 1024 -n 2000` | 16241 B | byte-identical |
| rollback harness (`rbtest`) | `cksum=4903597782947487587`, `rollback_replay_mismatch=0` | identical |
| `test-save-load-state -c 4096` | 7 PASS | 7 PASS |

Plus 0 re-reserves in the `-c 8192` perplexity run with the pool on (17 before the short-run fix).

## raw

Not committed, and no longer in the worktree: stashed on 2026-09-25 as
`stash@{0}: E056 raw output (gates, abc A/B/C, patch)`, whose untracked-files commit is
`4e9f11d67`. 632 KB. To read it back without disturbing the worktree:

```sh
git show 4e9f11d67 --stat
git checkout 4e9f11d67 -- .pi/agent/memory/experiments/results/E056-pooled-reserve
```

- `abc-sfa/` - the table above, `.out`/`.err` per arm per pass, plus `summary.txt` and `run-abc.sh`
- `abc/` - the same without sparse FA/rtile (pp ~640, which is the dense-FA level, not a regression)
- `gates/` - every gate above, and `re-reserve-census.txt`
- `../E056-pooled-reserve.patch` - the change as measured, now `b811339d6`
- T2, the user's run of the command block: `results/user/h9-d2-8be59fb/` (that directory has its own
  `.gitignore`, so it is local by construction)
- T2 motivation, the user's pre-existing logs, still in the worktree:
  `results/user/p2p-improvements/run8-sparse-fa.log` and `run8-sparse-fa-noprefillpool.log`

## verdict

**accepted, on T1 and T2.** The churn is gone on both boxes (0 re-reserves in every pooled arm), the
pooled reservation is *smaller* than the historic one at 131k, pp is at parity or better with the pool
off, tg keeps the pool's +32.6% at depth, and every numeric gate is bit-identical.
`Q4EXP_POOLED_NO_PREFILL` has no remaining job: arm C dominates arm B on both boxes (2 fewer
re-reserves per pass, +1.8% pp at 131k, same tg), and prefill now leaves the pool warm, so the first
decode step does not re-derive all 34816 blocks.

## T2 results (bench box, 4 cards)

Run by the user on 2026-09-25 from `8be59fb38` (the code of `b811339d6`): the command block below, 2
interleaved passes x `-r 3`, `-d 131072 -p 8192 -n 128 -ub 1024 -b 2048`, real Q4_K-M weights over 4
GPUs. Raw: `results/user/h9-d2-8be59fb/` - that directory carries its own `.gitignore`, so it is never
committed.

| arm | pp8192 @ d131072, pass 1 then pass 2 | tg128 | re-reserves / invocation | pp reservation |
| --- | --- | --- | --- | --- |
| A pool off | 1423.78 ± 7.69, 1425.67 ± 9.12 | 24.77, 24.69 | 0 | 2353.85 MiB, 6775/1358 |
| B `NO_PREFILL` | 1436.33 ± 8.10, 1428.98 ± 9.17 | 32.47, 32.71 | 2 (~57 ms) | 2353.85 MiB, 6775/1358 |
| C pooled (**new**) | 1454.65 ± 6.71, 1463.35 ± 5.80 | 32.85, 32.74 | **0** | **2319.04 MiB**, 6825/1372 |

- **The churn is gone on 4 cards.** C has no `sched:realloc_*` region in any table, against run8's 19-133
  per test for the same configuration and the same depth.
- **The prediction was right in delta and wrong in level.** Within run8, pooling prefill cost -13.4%
  (1299.60 vs 1501.24); within run9 it is +1.8% (C vs B). That 15-point swing is the churn. But arm B
  itself moved from 1501.24 to 1432.7 between the two sessions, so "recover to ~1500" was a
  cross-session claim and does not hold - only the within-session one does. A and B reserve identically
  (2353.85 MiB, 6775 nodes), so B's runtime prefill is untouched by this change and that shift is
  session noise, not the patch.
- **Pooled prefill at 131k is a small win, under the protocol's 5% bar:** C vs B +1.8%, C vs A +2.4%,
  from 6 alternating samples per arm. The ordering is clean (both C samples above both B above both A)
  and the within-arm spread is <= 0.6%, but at this magnitude it is suggestive, not a result.
- **Where the pp win sits:** `graph:set_inputs` is 90.8 ms/call in C against 125.9 in A on the pp table,
  because the pooled variant never creates `blk_pos` - I32 `[4*n_blocks*n_stream]`, 557 KB per ubatch at
  131k. That is E028/P2's prize, collected for free by the pool, and it is host-side, so it never shows
  up in a device trace.
- **The pooled reservation is 34.8 MiB smaller** than the historic one at this context (2319.04 vs
  2353.85), not larger: dropping `blk_pos` and the whole-cache rope outweighs the write chain. The dev
  box's +0.08 MiB is the same trade at a fifth of the context.
- **tg confirms why the pool exists:** +32.6% over pool off (32.80 vs 24.73 median), matching E045's
  +30.1%. B and C are the same workload in this benchmark (llama-bench drives no spec decode) and differ
  by 0.6%, i.e. nothing.

## notes

**What run8 said the churn costs on 4 cards.** Same session, same flags except the pool width gate:
pp8192 1780.78 (pooled prefill) vs 2229.91 (`NO_PREFILL`) at d4096, 1299.60 vs 1501.24 at d131072.
Converting to ms/ubatch (1024 tokens) gives +116 ms at d4096 and +106 ms at d131072 - a *fixed*
per-ubatch penalty, depth-independent, which is the signature of host cost rather than device work.
The region sums agree: for pp512 @ d131072, `alloc + compute` is 74.5 s pooled vs 60.4 s unpooled,
and the pooled arm's `graph:compute` is *smaller* (22.2 s vs 57.5 s) because the mandatory drain
moves into `graph:alloc`'s synchronize, exactly as the E049 review argued. The part that does not
move back is the lost host/device overlap: `graph:set_inputs` is ~95 ms/ubatch there, and that is
the size of the penalty. Predicted recovery on the bench box: pp8192 @ d4096 1780 -> ~2230, @ d131072
1300 -> ~1500. **That prediction was half right and is corrected by the T2 results above:** the delta
recovered (+15 points between the pooled and unpooled arms), but the absolute level did not, because
comparing run8 to run9 is a cross-session comparison and arm B alone moved -4.6% between them.

**Why E051's d2 attempt failed and this one did not.** It keyed the worst case on the *cell state*
(its record describes deleting the `nu == 0` test, which then asserted on `cells.pos_get()` of an
empty cache). This one keys it on the *context kind*: `llama_memory_hybrid_idx_context::is_update`,
the same flag that already fakes `ns_ubatch` for the reservation. A full-cache context has no ubatch
state by construction, so there is nothing to read and nothing to assert on.

**Pre-existing churn that is not the pool's.** `llama-cli -c 4096 -n 2500` on `q4exp-4l` re-reserves
**13 times with `Q4EXP_POOLED=0`** and 13 times with it on (2 of them "different number of nodes", so
this arch has another shape switch somewhere); `-c 1024 -n 2000` is 6 and 6. `llama-perplexity -b 256
-c 2048` is 120 and 120. At the ~300 ms a re-reserve costs on 4 cards that is seconds per session,
and it is a separate bug from H9. Tracked as **H19** in `plans/backlog.md`, which carries what I found
after writing this record: the growing tensor is `blk_cells` (named by its consumer,
`GET_ROWS(cache_idx_k_l3, ...)`), it steps once per 256 tokens because `llama_kv_cache::get_n_kv` pads to
`max(n_pad, 256)`, and the ten decode events are collateral from three prompt-phase mismatches that drop
the worst-case budget. Still unnamed: the two node-count changes.

**The recorded golden gate is vacuous for the pool.** `llama-perplexity` computes
`n_seq = max(1, n_batch/n_ctx)`, so the standard golden command (`-c 512 -b 2048`) runs **4 sequences
per batch** and `qsa_pool_get` returns NONE for all of them: 0 `qsa pool: mode` lines. Every record
that cites 263113.6984 as the pool's gate (E044, E049, E051, E052) was citing a pool-off run in both
arms. The gates that do engage it are the `-c 8192` sparse corpus (n_seq=1, 49 pooled calls), `-b 256
-c 2048` (121 calls) and the two `llama-cli` runs above.

**What still builds the historic shape, and therefore still churns.** The reservation can only hold
one shape, so any runtime state where `qsa_pool_get` answers NONE now mismatches it: more than one
sequence in the cells, a non-dense or non-contiguous run (an interior `seq_rm`), or a position past
the cell window. Reaching those needs `Q4EXP_POOLED` set *and* (`--kv-unified` with >1 slot, or a
cache with holes) - `qsa_pool_on` is `(unified || n_seq_max == 1)`, and llama-bench/llama-cli/single
-slot servers are always the fast state. Truncating rollbacks (MTP rejection) stay dense, so the
deployment in E045 is covered. The escape hatch is unchanged: `Q4EXP_POOLED=0`.

Closing that last gap means making the *general* path build the pooled topology too (`n_new =
n_blocks`, write lists filled from the group machinery, which already computes `bid_idx`/`bid_cell`
and the per-block positions). ~40 lines in `set_input_qsa`, and it would make the shape state-
independent. Not done: it costs the general path a pool write it cannot use, and it needs a
multi-sequence gate that does not exist yet.

## T2 command as run (kept for reproduction)

```sh
# same flags as run8, three arms, interleaved. -r 3 minimum, one depth per invocation
for pass in 1 2; do for arm in A B C; do
  case $arm in
    A) e="Q4EXP_POOLED=0";;
    B) e="Q4EXP_POOLED=1 Q4EXP_POOLED_NO_PREFILL=1";;
    C) e="Q4EXP_POOLED=1";;
  esac
  env $e GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_PROF_REGIONS=1 \
      GGML_FATTN_RDNA_RTILE=1 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_SPARSE_FA=1 \
    llama-bench -m <model> -ngl 99 -sm tensor -fa 1 -ot per_layer_token_embd=CPU -lzm on-direct \
      -d 131072 -p 8192 -n 128 -ub 1024 -b 2048 -r 3 > run9-$arm-p$pass.out 2> run9-$arm-p$pass.err
done; done
grep -c "sched:realloc" run9-C-*.err   # expect 0; run8 had 19-133 per test
```

Deciding metric: `sched:realloc_*` counts (expect 0 in C) and pp8192 C vs B. Answered above: 0 in C,
and C is +1.8% over B at d131072, so `Q4EXP_POOLED_NO_PREFILL` can go.
