# E042 - decode-only trace pair after H13 fix: the chain is data movement, not top_k

> Corrected by E043 (reviewer-5). The totals and the mechanism below changed; see the inline edits and
> E043 for the full review. Headlines that stand: top_k is negligible (0.55), and ~11.1 of the 31.9
> ms/token/GPU of the arm is recomputing the static pooled indexer block keys - which is cacheable
> without changing outputs (H9, output-neutral, reference-equivalent).

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
| indexer whole-cache gather f16->f32 (all 131072 cells) | 3.00 | ~0 | +3.00 |
| 4 pooling rect copies (cont of strided views) | 3.00 | 0.05 | +2.95 |
| rope of all 32768 pooled block keys | 2.55 | 0.03 | +2.52 |
| index_k_norm over 32768 rows | 1.09 | - | +1.09 |
| k_bin_bcast add family total | 1.48 | ~0.09 | +1.39 |
| top_k trio (hist/select/gather) | 0.55 | - | +0.55 |
| score matvec + rest of the query-dependent part | ~1.5 | ~0.2 | +1.3 |
| model matvecs (mmvq 6/8/12/14, mmvf) | 11.2 | 11.0 | +0.2 |
| rest | ~0.4 | ~0.2 | +0.2 |
| total device time (sum of ALL log rows) | **31.9** | **21.3** | **+10.5** |

- Decode wall at 131k is 43.2 ms/token (23.16 t/s), so the qsa arm is ~74% busy (31.9/43.2), corrected
  from 56%: the first version summed its own table, not the logs, and missed ~7.8 ms/token/GPU.
  Non-kernel wall is ~11 ms/token, so device-time savings convert to wall at close to 1:1.
- Same call count in both arms for nccl (147k), but in the no-qsa arm it is 28% longer per call.
- `flash_attn_rtile` on the qsa arm: 9216 calls, 110.98 ms total, 12.04 us/call - but see E043 F8: the
  FA-family calls are 6/token/GPU, not the expected 12 = per-agent grouping is unresolved; treat the FA
  rows as lower bounds of up to 2x.

## What kind of answer this is

Before, H13 was measured by hunches. This is the first decode-only attribution on the real model:
- rtile: done, it shows.
- **top_k now costs 0.55 ms/token/GPU (~5% of the added cost). H13 did its job.**
- The corrected mechanism (E043 F2-F4): the 10.5 ms is NOT materializing a 2052-cell K/V slice for the
  core attention - nothing does that; FA reads cache rows directly through the compaction indices, and
  the main K is roped once at write time. The cost is the INDEXER re-deriving all 32768 pooled block
  keys per token: whole-cache gather (2.99) + 4 pooling rect copies (2.95) + pooled rope (2.50) +
  pooling adds (1.27) + scale (0.27) + index_k_norm (1.09) = ~11.1 ms/token/GPU. Every input to the
  block key (raw indexer keys, block start positions, norm weight) is static once a block is complete;
  only top_k selection is query-dependent (~2.1 ms). So 84% of the chain is redundant recomputation,
  and caching it is exactly the backlog's H9 - arithmetic-identical, output-neutral (the earlier
  "changes output" claim here was wrong; that is the MTP trick, a different idea).
- Also first measurement of the both-arms common floor: **NCCL 4.6-5.9 ms/token/GPU alone**. It is the
  biggest single term in every arm, means allreduce of the split-mode matvecs over PCIe with no NVLink.
  In the qsa arm it accounts for ~19% of device time; in the comparison, no-qsa pays even more.

## Comparison to previous interpretation (E036-E038, pre-H13)

- top_k demoted from biggest chain term (3.66% full-run share then) to 0.55 ms/token; the earlier
  dominant terms (expands, copies, mask-add) are gone, per E041's block-selection table.
- E037's H9 bullet is wrong ("rms_norm and rope stay, since they are position-dependent"): block keys use
  the block START position, which never moves - so the whole gather+pool+norm+rope family is cacheable
  and H9's ceiling is the full 11.1 ms, not ~3.4 s of ~14.8 s.
- The reviewer also found the user's hunch (a) is half right: there is no 4096-cell recent-window fast
  cache; the always-included set is just the r=4 tail. The cache that makes sense is the pooled block
  keys (1/4 the raw cache size), which is H9. Main-attention K/V are already roped-at-write and are not
  re-gathered per token.
- Two cheap launch defects (E043 F7): rope_multi runs 64 of 256 threads on the 128-wide shape
  (163 GB/s vs the 400-550 GB/s neighbours, ~1.7 ms/token/GPU ceiling), and mask_to_sparse_indices
  launches one 256-thread workgroup per decode mask (2.7 GB/s, ~0.4 ms recoverable). Both generic,
  both small diffs.

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
- H9 first (E043): see runs/E043-review-of-e042.md. Fuse gather+rope+adds is second and is a bespoke
  kernel, not the fusion matcher (mrope mode gate + node adjacency block all useful patterns).
- E043 F8: FA-family kernels are 6/token/GPU, half the expected 12 - disambiguate with per-agent
  grouping of a kernel-trace CSV before quoting FA rows.