# E037 - what the QSA chain actually costs, per kernel (bench, real model)

Source: `results/user/qsa-kernel-traces/stats-qsa.log` and `stats-no-qsa.log` - per-kernel aggregate
statistics the user pulled on the 4x R9700 box with the real checkpoint, two arms differing only by
`Q4EXP_NO_INDEXER=1`. rtile was **off**, so FA stayed on the vec path in both arms (E034's dense/sparse
gate comparison in E027 was the same situation). This is the first per-kernel evidence on the real
model, and it retires two of my claims the same day.

## Totals

| | chain on | chain off | delta |
| --- | --- | --- | --- |
| kernel ms | 113,315 | 98,540 | **+14,775 (+15.0%)** |
| kernel launches | 6,508,352 | 5,663,612 | **+844,740** |
| FA (`flash_attn_ext_f16` + tile) | 5,856 / 20,400 | 5,969 / 20,400 | -113, **same calls** |
| NCCL (`ncclDevKernel_Generic_4`) | 22,036 / 163,124 | 22,212 / 163,124 | -176, same calls |

Identical FA work confirms the vec path was used: with vec, the selection only changes mask *values*,
so sparse buys nothing at attention and the chain is pure cost - which makes the delta a clean read of
the chain. 1,700 chain invocations (`fill_kernel<__half>` calls / 12 QSA layers), of which 320 are
prefill ubatches and 1,380 decode steps.

**844,740 extra launches / 1,380 decode steps = 612 launches per token.** The measured wall-clock gain
from killing the chain was 15.5 ms/token (E036 follow-up: 22.24 -> 33.89 t/s at 128k), so ~25 us per
launch of host dispatch explains the whole effect. The chain's *device* time is 14.8 s / 1,380 = 10.7
ms/token summed over four cards, so about 2.7 ms/token of wall clock if evenly split. The gap between
2.7 and 15.5 is dispatch, not bytes.

## The chain, ranked (delta = on minus off)

| item | delta ms | extra launches | us/call | note |
| --- | --- | --- | --- | --- |
| `top_k_radix_*` | 3,121 | 223,344 | 11 launches per call | GGML_OP_TOP_K is init + 4x(histogram, select) + reset + gather |
| `k_get_rows_float` | 2,907 | 40,160 | 73 (f16 src) / 12 (f32) | the indexer gather and the cell expand, 2 per layer-step |
| `k_bin_bcast` (add) | 1,379 | 101,360 | 14 | 5 per layer-step: 3 pooling adds, score and mask adds |
| `rope_multi<f32>` | 1,344 | 40,800 | 33 | 2 per layer-step |
| `cpy_scalar*`/`cpy_*_contiguous` | 1,182 | 24,240 | 191 (prefill transpose) | the `cont(permute())` pair |
| `copyBufferRectAligned` | 909 | 83,616 | 11 | 4 per layer-step: the stride-`r` pooling slices |
| `mul_mat_vec*` | 782 | 55,440 | 14 | indexer q/k projections and the score product |
| `rms_norm_f32` | 585 | 40,800 | 14 | 2 per layer-step |
| `unary_op<relu>` | 497 | 20,400 | 25 | 1 per layer-step over the f32 score array |
| scale, fill, set_rows, convert | 558 | 184k | 1-7 | mask build and the mean scale |

Per QSA layer per step that is ~41 launches and ~725 us of device time, against ~69 launches per layer
for everything else in the model. So the chain is not one expensive kernel; it is 41 small dependent
ones, 11 of which belong to `top_k` alone.

## What each planned fix can remove

- **H13 (select at block level, then expand) - ~6.8 s, 46% of the chain.** It shrinks `top_k`'s input
  4x, deletes the cell-level expand (`k_get_rows_float<float,float>`, 1.16 s), the per-cell `relu`
  (0.50 s), the `cont(permute)` pair (1.18 s) and part of the adds. At their scale this is the biggest
  single lever, which is the opposite of what E036's follow-up concluded from the dev box.
- **H9 (pooled indexer keys) - ~3.4 s, 23%.** The key gather (1.50 s), the four strided slices (0.91),
  the pooling adds (~0.9) and the scale (0.15). `rms_norm` and `rope` stay, since they are
  position-dependent and re-applied per step.
- **Not addressed by either - ~2.9 s, 20%:** rope 1.34, rms_norm 0.59, the indexer matmuls 0.78.

## Corrections to earlier records

- "TOP_K costs ~88 us flat, so it is a fixed cost and not the problem" (E036 follow-up) was measured on
  the dev box with 4x fewer elements and *four fewer* radix passes of a smaller array. On the real model
  it is 153 us per call across 11 launches, and it is the **largest** chain item at 3.1 s. H13 stays; the
  demotion is retracted.
- H13's "bit-identical" acceptance test stays wrong for a different reason than E036 gave: the tail is
  force-included by scoring tail blocks at `+1e9` (`llama-memory-hybrid-idx.cpp:635`) and the width
  `2048 + r - 1` is not a multiple of `r`, so today's cell-level selection can split a block where
  block-level selection cannot. Validate by deep-arm PPL plus a set-overlap count, not by equality.
- The gather's destination is **f32 from an f16 cache** (`k_get_rows_float<__half, float>`), so the
  pooling then runs on f32 arrays at 2x the bytes. A dtype-side change (keep the members and pooling in
  f16) is worth ~0.5-1 s of this on its own and needs no new op.

## Questions this raises (unanswered)

- Were both arms run with CUDA graphs off (`-dgs none` or a `-ncg` build)? If so the 612-launch/token
  chain is fully exposed, and re-running the same two arms with graphs on is the decisive test of the
  dispatch hypothesis. There are `results-no-cudagraphs.csv.log` and
  `results-disable-cudagraphs-reuse.csv.log` in the user directory, so graphs appear to be a known
  problem on their fork - which would make "get graphs working" the top item on the backlog, above H13.
- NCCL is 22 s in *both* arms (19% of all device time, 96 calls per invocation at 135 us each, i.e. 2
  per layer per step, mostly peer-wait). Identical in both arms, so the chain is not causing it; it is a
  separate 4-card item worth its own look.
- Whether the chain's kernels run once per device over a quarter of the cells. The aggregates cannot
  say (no agent column here); the full traces would. If they do, the selection is per-device and the
  union of four local top-k's is not the global one - a correctness issue, and `minimax-m3` guards
  exactly this with a `unified()` assert while `llama_memory_hybrid_idx` does not.
