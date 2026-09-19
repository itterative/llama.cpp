# E037 - what the QSA chain actually costs, per kernel (bench, real model)

Source: `results/user/qsa-kernel-traces/stats-qsa.log` and `stats-no-qsa.log` - per-kernel aggregate
statistics the user pulled on the 4x R9700 box with the real checkpoint, two arms differing only by
`Q4EXP_NO_INDEXER=1` (build `7d376d617` plus `tools/qsa-no-indexer.patch`). rtile was **off**, so FA stayed on the vec path in both arms (E034's dense/sparse
gate comparison in E027 was the same situation). This is the first per-kernel evidence on the real
model, and it retires two of my claims the same day.

> **Trust ratios, not absolute ms:** the stats table's agent coverage and whether `--stats`
> replays kernels are unconfirmed, so absolute millisecond figures are unknown up to a
> constant. E038's cache-placement section supplies the wall-clock bound that keeps the
> per-card reading honest (1.5x tg from dropping the chain rules out 4x replication).

## Totals

| | chain on | chain off | delta |
| --- | --- | --- | --- |
| kernel ms | 113,315 | 98,540 | **+14,775 (+15.0%)** |
| kernel launches | 6,508,352 | 5,663,612 | **+844,740** |
| FA (`flash_attn_ext_f16` + tile) | 5,856 / 20,400 | 5,969 / 20,400 | -113, **same calls** |
| NCCL (`ncclDevKernel_Generic_4`) | 22,036 / 163,124 | 22,212 / 163,124 | -176, same calls |

Identical FA work confirms the vec path was used: with vec, the selection only changes mask *values*,
so sparse buys nothing at attention and the chain is pure cost - which makes the delta a clean read of
the chain. 1,700 chain invocations (`fill_kernel<__half>` calls / 12 QSA layers); 320
invocations include a node that only prefill builds (`cpy_scalar_transpose`, 3,840 calls = 320 per
layer at 191 us), and 3 x 128 = 384 tokens were decoded, so the split is not yet known - see "Run shape".

What is shape-independent: **497 extra launches per chain invocation = 41 per QSA layer**, and 14.775 s of
device time over 1,700 invocations = **8.7 ms per invocation = 0.72 ms per QSA layer per invocation**. The
average chain kernel is 17.5 us of device time, which is small-kernel territory either way.

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

## Run shape, and the numbers that depend on it

Command (build `7d376d617` + `tools/qsa-no-indexer.patch`, so the two arms differ *only* by the chain):

```
rocprofv3 --kernel-trace --stats --output-format csv -o result_qsa_decode.csv -- \
  llama-bench -m <Qwen3.8-Flash-Next-Q4_K_M> -lm none -sm tensor -fa 1 -lzm on-direct \
  -ot per_layer_token_embd=CPU -d 40960 -p 0 -n 128 -r 3 -b 2048 -ub 1024
```

Shape-independent facts: 1,700 chain invocations over 12 QSA layers, so 497 extra launches per
invocation = **41 per QSA layer per invocation**, and 8.7 ms of device time per invocation = **0.72 ms
per QSA layer**. But 1,700 invocations against `3 x 128 = 384` decode tokens means the chain runs about
4.4x per token, so either `-d 40960 -p 0` prefills to depth (120 ubatches at `ub=1024`) *and* the fork
runs more than one graph pass per token, or both. Until that is settled, ms-per-*token* claims from
these aggregates are unreliable; the ranking and the 46/23/20% grouping are per-invocation and stand
either way.

## Two side results

**The indexer cache is f16, proven.** `llama-perplexity -v` on the dummy prints
`llama_kv_cache: size = 0.50 MiB (512 cells, 1 layers, 4/4 seqs), K (f16): 0.50 MiB, V (f16): 0.00 MiB`
immediately after `creating indexer KV cache` - so E036's correction this morning holds: `-ctk` drives
both caches and the cached indexer key is 256 B per token. Method notes: `HIP_PROFILE=1` prints nothing
on the dev box, and piping a `timeout`-killed run into grep loses the block-buffered log (a trap already
in the records) - redirect to a file first.

**One row the aggregates cannot name.** `k_get_rows_float<__half, float>` has a delta of exactly
+20,400 calls (1 per QSA layer per invocation) at 73 us each, 1.50 s. `getrows.cu:71` is
`template<typename src0_t, typename dst_t>` and `ggml_cuda_op_get_rows` dispatches on `dst->type`
(`getrows.cu:413,457`), so this is f16 in, f32 out - which a plain `get_rows` of an f16 cache cannot
produce. Candidates: `members`, if the pooling really is staged through f32 (weakly supported by five
f32 `op_add` instances per layer-step), or the argsort path in `ggml-cuda.cu:2077` - but
`k_argsort_f32_i32` appears in *both* arms, so not that. If it is `members`, f16 staging is worth ~1.5 s
plus a slice of the adds: backlog H15, no new op. Needs naming the node, not more aggregates.

## What it would take to answer the rest

- A third arm at `-d 131072`, same command otherwise: terms that scale with depth (gather, pooling,
  expand) separate from terms that do not (top_k's 11 launches, the mask fill), and 131k is where the
  33.89 t/s was measured.
- Whether the fork runs extra graph passes per token, and whether `-d` prefill-seek is real, to convert
  per-invocation into per-token.
- The full `--kernel-trace` CSV only if it carries a device/agent column - that is the one question
  aggregates cannot reach: does the chain run once per device over a quarter of the cells?

## Questions this raises (unanswered)

- NCCL is 22 s in *both* arms (19% of all device time, 96 calls per invocation at 135 us each, i.e. 2
  per layer per step, mostly peer-wait). Identical in both arms, so the chain is not causing it; it is a
  separate 4-card item worth its own look.
- Whether the chain's kernels run once per device over a quarter of the cells. The aggregates cannot
  say (no agent column here); the full traces would. If they do, the selection is per-device and the
  union of four local top-k's is not the global one - a correctness issue, and `minimax-m3` guards
  exactly this with a `unified()` assert while `llama_memory_hybrid_idx` does not.
