# E042 - decode-only trace pair after H13 fix: the chain is data movement, not top_k

`results/user/qsa-kernel-traces-postfix/stats-*-rtile-131k-regions.log`, captured with `rocprofv3
--selected-regions` around `test_gen` (the E041-era llama-bench patch, `--marker-trace` required for the
tool to honor the regions). Grid: d131072, p0, n128, r3, both arms `-sm tensor -fa 1 -lzm on-direct`.
The 131072-token depth fill is excluded; only ~385 decode tokens (3x128 + 1 warmup) are in the capture,
on 4 GPUs. All numbers below are ms per token per GPU.

| kernel/family | qsa | no-qsa | delta |
| --- | --- | --- | --- |
| ncclDevKernel_Generic_4 (allreduce) | 4.59 | 5.89 | -1.30 |
| dense FA (rtile `<...,false>` over 131072) | - | 2.03 | -2.03 |
| sparse FA: mask_to_sparse_indices + rtile | 0.64 | - | +0.64 |
| k_get_rows_float<half> (K/V cell gather) | 3.00 | ~0 | +3.00 |
| __amd_rocclr_copyBufferRectAligned | 3.00 | 0.05 | +2.95 |
| rope_multi | 2.55 | 0.03 | +2.52 |
| k_bin_bcast add | 1.69 | 0.09 | +1.60 |
| rms_norm<256> | 1.09 | 0.14 | +0.95 |
| top_k trio (hist/select/gather) | 0.53 | - | +0.53 |
| model matvecs (mmvq 6/8/12/14, mmvf) | 7.15 | 7.00 | +0.15 |
| rest | ~0.5 | ~0.2 | +0.3 |
| total device time | 24.1 | 15.3 | +8.7 |

- Decode wall at 131k is 43.2 ms/token (23.16 t/s), so the qsa arm is ~56% busy (24.1/43.2). 
- Same call count in both arms for nccl (147k), but in the no-qsa arm it is 28% longer per call.
- `flash_attn_rtile` on the qsa arm: 9216 calls, 110.98 ms total, 12.04 us/call - the 2052-wide sparse
  attention is negligible (0.07 ms/token/GPU).

## What kind of answer this is

Before, H13 was measured by hunches. This is the first decode-only attribution on the real model:
- rtile: done, it shows.
- **top_k now costs 0.5 ms/token/GPU (7% of the qsa's added cost). H13 did its job.**
- The meat of the added 8.7 ms/token is data movement: the K/V cell gather (3.0) plus its staging
  copies (3.0) plus rope of the selected cells (2.6) - the stuff that materializes the 2048-selected
  cell slice per token in every QSA layer. Lower bound by memory traffic: gather reads 2048 cells f16 per
  layer; rope/scale/adds touch it 3 times with no kernel fusion.
- Also first measurement of the both-arms common floor: **NCCL 4.6-5.9 ms/token/GPU alone**. It is the
  biggest single term in every arm, means allreduce of the split-mode matvecs over PCIe with no NVLink.
  In the qsa arm it accounts for ~19% of device time; in the comparison, no-qsa pays even more.

## Comparison to previous interpretation (E036-E038, pre-H13)

- top_k demoted from biggest chain term (3.66% full-run share then) to 0.53 ms/token; the earlier
  dominant terms (expands, copies, mask-add) are gone, per E041's block-selection table.
- What survived H13: the `n_sel` gather/rope/adds path, which the old grouping would have called
  "36% removable by H9". This trace gives the reason that number matters: it's now 85% of the chain.
- The chain cost is now dominated by per-token re-materialization of the selected cells. To cut it,
  either more pooling (H9-esque reuse across steps) or bigger work per kernel (fuse gather+rope+adds,
  pack K/V read as 2D to cut copyBufferRectAligned).

## Caveats

- The tool capture excludes the depth fill. The 1/128 warmup decode is inside the region.
- These numbers are all averaged per token per GPU, assuming the 4 GPUs split evenly (fine for ranking).
- The kernel names are demangled/truncated by rocprofv3 (`-T`), but all above resolve trivially.

## Follow-ups

- qsa wall: 23.16 t/s unprofiled (results-h13-post-review.log). no-qsa wall: 24.29 ± 0.69 t/s under rocprofv3
  (much slower than unprofiled; the qsa arm keeps hanging the GPU under rocprofv3, so a clean profiled
  pair is not obtainable). To finish the pair, one plain unprofiled
  `Q4EXP_NO_INDEXER=1 llama-bench -p 0 -n 128 -d 131072` run - no profiler, no hangs. Meanwhile 15.3
  ms/GPU of device time vs 41.2 ms/token wall = 37% busy for no-qsa under the tool, so both arms spend
  most of the wall outside kernels.
- NCCL: dropped as an action item - fixed comm tax of 4-GPU split mode, user is right there is nothing
  to do there. The no-qsa arm's longer per-call NCCL time at equal call count is queueing behind dense
  FA, not an NCCL configuration issue.
- Fuse gather+rope+adds into one kernel, or check `copyBufferRectAligned`'s caller. Biggest single
  lever left on the decoder side.
- H9 (skip-and-reuse across steps) can cut the whole materialization family at once, but changes output.