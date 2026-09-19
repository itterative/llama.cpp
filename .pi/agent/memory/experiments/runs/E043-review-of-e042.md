# E043 - reviewer-5 on E042: the chain is indexer recomputation, and H9 is output-neutral with a measured 11.1 ms ceiling

Background review of E042 (decode-only kernel traces). Read-only; no builds, no runs. Line refs are
`src/models/qwen4exp.cpp` unless stated. Summary of findings in priority order; see the review for
per-kernel maps and the fusion analysis.

## The correction that changes the plan: E042's mechanism was wrong

- The cost is NOT "materializing the 2048-selected cell slice". Nothing does that. FA reads cache rows
  directly through compaction indices; the main K is roped once at write time (`:891` -> `:785`) and
  the cache holds roped keys.
- The 10.5 ms/token/GPU delta [log-sum corrected from E042's 8.7: qsa 31.9 vs no-qsa 21.3; E042's own
  24.1/15.3 omitted ~7.8 ms of rows, and qsa is ~74% busy, not 56%. Non-kernel wall ~11 ms/token, so
  device-time wins convert to wall at ~1:1.]
- The cost is the indexer re-deriving all 32768 pooled block keys per token per QSA layer:
  whole-cache f16->f32 gather `:638` (2.99) + 4 rect copies from `cont(view_3d)` at `:643-647` (2.95) +
  pooled rope at `:658` (2.50) + pooling adds `:647` (1.27) + scale (0.27) + index_k_norm `:654` (1.09).
  Gating details: `__amd_rocclr_copyBufferRectAligned` IS the `ggml_cont` fast path - `GGML_OP_CONT`
  -> `ggml_cuda_dup` -> `cpy.cu:474-478` `hipMemcpy2DAsync`, 4 per QSA layer (r), 546 GB/s - they are
  efficient copies that exist only because `ggml_add` wants contiguous rows.
- The block key is rope_p_b(rmsnorm(mean(raw_k[cell(b,i)])) * w_norm): every input is static once the
  block completes (raw keys written once `:626-631`; the rope position is the block START
  `llama-memory-hybrid-idx.cpp:503-505`; norm weight constant). Only top_k selection is query-dependent
  (~2.1 ms of the ~13.2 chain). So 84% is redundant recomputation: one new block per 4 tokens out of
  32768, re-derived 131072x too often, x4 again because the cache is mirrored.

## Consequences for the records

- E037's H9 bullet ("rms_norm and rope stay, position-dependent") is wrong: block positions never
  move, so H9's prize is the full 11.1 ms, not ~3.4 s of ~14.8 s. Corrected in E037 and E042.
- E042's "H9 changes output" was wrong: pooling block keys at write time is arithmetic-identical
  (the reference does this; report "key compression before positional encoding", block start p_b).
  Output-changing is only the MTP trick (reuse selection across spec-decode steps, report Table 4).
- The user's hunch about a ~4096-cell fast cache: half right. No recent-cell K/V fast cache exists;
  the always-included set is just the r=4 tail. The cache that makes sense is the POOLED BLOCK KEYS,
  1/4 the raw cache size. Main-attention K/V are not re-gathered per token.
- Not qwen4exp-specific: `src/models/deepseek4.cpp:471-517` builds compressed KV from state the same
  way (gather + weighted reduce + norm + rope per step) - inherited pattern, an H9-style fix is
  arguably upstreamable.
- Backlog H4b "tg flat": pre-rtile-era note; with block selection + rtile the sparse arm pays ~1.4
  ms/token/GPU at decode (0.64 vs 2.03 dense rtile, 28x per call).
- F8, unresolved: FA-family kernels are 6/token/GPU in both arms, half the 12-QSA-layer expectation.
  Need a per-agent grouping of the CSV (rtile vs fill_kernel<half> as the 12-layer control); until
  then treat FA rows as lower bounds of up to 2x.
- F10: k_bin_bcast adds are +1.48 as a family, not +1.60 (rows are not pairwise comparable; truncated
  names hide template args).

## Two launch-config defects (cheap, generic)

- rope_multi on the 128-wide pooled keys: 64 of 256 threads work (grid 1 row per workgroup,
  rope.cu:472-476), 163 GB/s vs 400-550 GB/s neighbours. Ceiling ~1.7-1.9 ms/token/GPU.
- mask_to_sparse_indices: one 256-thread workgroup scans the 131072-entry decode mask
  (fattn.cu:102-118), 95.71 us for 262 KB = 2.7 GB/s. Chunk/prefix version recovers ~0.4 ms.

## Why the fusion matcher will not reach this graph

The in-tree machinery is real ({RMS_NORM,MUL,ROPE[+VIEW+SET_ROWS]} -> rms_norm_mul_rope_f32,
{ROPE,VIEW,SET_ROWS} -> rope-into-KV-cache, {RMS_NORM,MUL[,ADD]} -> rms_norm_f32<...,true> (this one
fires here), FFN gate/up, SSM, topk_moe, gated_delta_net). It misses qwen4exp twice:
the rope-mode gate (`ggml_cuda.cu:2763-2767`, `:2803-2807` require NORMAL/NEOX; qwen4exp uses MROPE),
and node adjacency (norm `:654` and rope `:658` are separated by reshapes, `ggml_can_fuse_subgraph_ext`
matches consecutive indices). A bespoke fused kernel (gather+pool+norm+rope) is a realistic ~200-300
line HIP kernel with in-tree precedent (topk_moe is the structural analogue), but note it is a SYSTEM
for computing a value that should be CACHED. Do H9 first; the fused kernel then only runs on the
completion of 1 block per 4 tokens.

## Recommendation, as numbered by the review

1. **H9 - cache pooled indexer block keys at write time, persistent per-layer tensor `[128, n_blocks, 1]`**
   (f16 or f32, mirrored like the indexer cache). Host knows the newly completed blocks exactly
   (bid_idx/n_bid idioms already exist in llama-memory-hybrid-idx.cpp). Fast-path win: delete
   `:638-661` in the graph (the mapping is affine, no get_rows at all), score matvec `:674` reads the
   pooled cache directly (halves again if f16). Invalidation is the known cost: seq_add/div/rm/cp,
   defrag, clear, state save-load, plus a "dirty from block B" watermark with today's code path as the
   full-recompute fallback. Ceiling: 11.1 ms/token/GPU = 35% of the arm's device time.
2. The two launch-config fixes (F7) - together ~2 ms, independent of everything.
3. The bespoke fused kernel - only after H9.

## Review scope notes

- Reviewer did not build, did not run; all line refs cited, counts checked against the logs.
- F0 sanity: divisor 1540 = 385 tokens x 4 GPUs confirmed five independent ways (gated_delta_net 36,
  topk_moe 48, hc_pre 96, nccl 96, QSA-only kernels 12 each); block_count = 48, 12 QSA layers.