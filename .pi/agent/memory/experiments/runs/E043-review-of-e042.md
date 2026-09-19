# Review: E042 decode-only QSA attribution (branch `experiments/qwen4exp-rdna4`)

Read-only review. No builds, no runs, no edits. All per-token-per-GPU figures below use E042's divisor (385 tokens x 4 GPUs = 1540), which I independently confirmed from the logs (see F0).

---

## PART 1 - FINDINGS, priority order

### F0. The logs sum to 4 GPUs x 385 tokens, and the layer structure is 48 / 36 GDN / 12 QSA

Before anything else, the divisor. Four independent anchors in `stats-qsa-rtile-131k-regions.log` all give exactly 1540 token-GPU units:

| kernel | calls | /1540 | meaning |
| --- | --- | --- | --- |
| `gated_delta_net_cuda<128,false,false>` | 55440 | 36.0 | 36 linear-attn layers |
| `topk_moe_cuda<512,false>` | 73920 | 48.0 | 48 MoE routers = 48 layers |
| `dsv4_hc_pre_f32<true>` | 147840 | 96.0 | 2 hyper-connection mixes per layer |
| `ncclDevKernel_Generic_4` | 147840 | 96.0 | 2 allreduces per layer |

Matches the GGUF: `block_count = 48`, `full_attention_interval = 4` (`.pi/agent/memory/experiments/results/user/gguf-dump.log:15,34`), so 12 full-attention/QSA layers and 36 GDN layers (`src/models/qwen4exp.cpp:127-135`). Confirmed a fifth time by the QSA-only kernels, all at exactly 12/token/GPU: `fill_kernel<__half>` (18480), `fill_kernel<float>` (18480), `k_get_rows_float<__half,float>` (18480), `top_k_radix_gather` (18432), `scale_f32` delta (18480), `mul_mat_vec_f<float,float,4,64>` (18480), `unary_op<relu>` (18480). So the stats file aggregates all four agents and E042's per-token normalization is right.

### F1. E042's "total device time" is the sum of its own table, not of the logs. The real totals are 31.9 and 21.3, and the qsa arm is ~74% busy, not 56%

Summing every row of each regions log:

| | sum of all kernel ms | /1540 = ms/token/GPU | E042 says |
| --- | --- | --- | --- |
| qsa | 48957.1 | **31.87** | 24.1 |
| no-qsa | 32761.8 | **21.33** | 15.3 |
| delta | 16195.3 | **10.54** | 8.7 |

E042's table omits ~7.8 ms/token/GPU in the qsa arm (`scale_f32` 0.66, `quantize_q8_1` 0.61, `mmvq<...,true>` 0.86, `mmvf<bf16,1,256>` 0.91, `rms_norm<1024>` 0.24, `hc_pre/hc_post` 0.31, sigmoid/silu/reduction/gdn/copyBuffer/... ), and its "model matvecs 7.15" undercounts the matvec family, which totals 11.2.

Consequence, and this is the part that matters for planning: E042 concludes "both arms spend most of the wall outside kernels" (56% and 37% busy). Corrected: **31.87/43.2 = 74% busy for qsa**, 21.33/41.2 = 52% for no-qsa. There is ~11 ms/token of non-kernel wall, not ~19. Device-side work therefore has roughly 2x the headroom E042 assumes, and a device-time saving of X ms converts to close to X ms of wall.

### F2. E042's central mechanism claim is wrong: nothing "materializes the 2048-selected cell slice". The cost is the indexer re-deriving *all 32768 block keys* every token

E042 says: "k_get_rows_float<half> 3.0 (K/V cell gather from the cache)", "rope of the selected cells (2.6)", "the stuff that materializes the 2048-selected cell slice per token in every QSA layer", and "Lower bound by memory traffic: gather reads 2048 cells f16 per layer".

All four statements are wrong, and they point the fix in the wrong direction.

* `k_get_rows_float<__half,float>` is **not** the main K/V cache and **not** a subset gather. It is `src/models/qwen4exp.cpp:638`, `members = ggml_get_rows(k_all, inp->blk_cells)`, where `k_all` is the **indexer** cache viewed as `[idx_dim, n_kv, n_stream]` (`:634-635`) and `blk_cells` is `I32 [r*n_blocks, n_stream]` (`:604`) = **all 131072 cells**. It is a whole-cache, f16->f32-widening, block-major permutation. Proof from the trace: 248.74 us per call. 2048 cells x 128 dims x 2 B = 0.5 MB would take ~2 us; 131072 cells x (2 B read + 4 B written) = 100 MB at 248.74 us = 402 GB/s, exactly a plausible DRAM rate. The row is also *absent* from the no-qsa log, which is what `Q4EXP_NO_INDEXER=1` removes.
* The 2.55 ms of `rope_multi` is **not** rope of selected cells. The main K is roped once, at write time (`src/models/qwen4exp.cpp:891` then `cpy_k` at `:785`), and the cache holds roped keys. The big rope is `:658`, `ggml_rope_multi(pooled, inp->blk_pos, ...)` over `[128, 1, n_blocks*n_stream]` = **all 32768 pooled block keys**, every token, every QSA layer.
* Nothing materializes a 2052-cell K/V slice anywhere. The sparse FA gathers rows inside the kernel from a compaction index buffer (`ggml/src/ggml-cuda/fattn.cu:20-100`, dispatched at `fattn-common.cuh:1095-1103`). That part is cheap and correct: `flash_attn_rtile<...,true>` is 12.04 us/call vs 337.48 us for the dense `<...,false>`, a 28x win, 0.07 ms/token/GPU.

So the ~8.7 (really ~10.5) ms is **selection-side**, not attention-side, and it is O(n_kv) per QSA layer per token rather than O(2048).

### F3. The real per-QSA-layer bill, and 84% of it is recomputation of a quantity that is static

Per token per GPU, from the two regions logs (deltas where the row exists in both arms):

| step | node(s) | ms/token/GPU | calls/token/GPU | us/call |
| --- | --- | --- | --- | --- |
| indexer whole-cache gather f16->f32 | `qwen4exp.cpp:638` | 2.99 | 12 | 248.7 |
| 4 pooling `cont(view_3d)` rect copies | `:643-647` | 2.95 | 48 | 61.5 |
| pooled rope over 32768 blocks | `:658` | ~2.50 | 12 (+12 tiny q ropes) | ~209 |
| adds (family total, incl. f16 mask add) | `:647`, `:687`, `:692`, `:815` | 1.48 | - | - |
| `index_k_norm` over 32768 rows | `:654` | ~1.00 | 12 (+12 tiny q norms) | ~85 |
| pooling `scale(1/r)` | `:649` | 0.27 | 12 | 22.4 |
| score matvec `[128,32768]x[128,4]` | `:674` | 0.59 | 12 | 49.0 |
| top_k (init+4x hist/select+reset+gather) | `:705` | 0.55 | 132 | - |
| mask compaction scan | `fattn.cu:20` | 0.57 | 6 | 95.7 |
| mask fill + set_rows + f16 add | `:792,:804,:808,:815` | 0.10 | 36 | - |
| relu / cells gather / concat / sparse FA / indexer cpy_k | `:677,:708,:715` | 0.19 | - | - |
| **total QSA chain** | | **~13.2** | | |

Split by whether the work is required by the QSA math:

* **Required (query-dependent): ~2.1 ms/token/GPU** - score matvec, relu, head sums, bias add, top_k, expand+concat, mask build, compaction, sparse FA.
* **Redundant (static, recomputed): ~11.1 ms/token/GPU** - gather + 4 rect copies + 3 pooling adds + scale + `index_k_norm` + pooled rope. This is 84% of the chain and 35% of the arm's entire 31.87 ms of device time.

### F4. THE KEY ANSWER: the redundancy is not inherent. The block key is a pure function of already-immutable data

The block key is `rope_{p_b}( rmsnorm( mean_{i<r}( raw_k[cell(b,i)] ) ) * w_norm )`.

Every input to it is static once the block is complete:

* The raw indexer keys are written once and never modified: `qwen4exp.cpp:626-631`, with the comment "cached indexer keys are raw: pooling precedes norm and rotation, so apply neither", and `src/llama-memory-hybrid-idx.cpp:54-56` which sets `hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE` for exactly that reason. The indexer cache allocates no V at all (`:57-60`).
* The rope position is the block's **start position**, not a per-token quantity: `llama-memory-hybrid-idx.cpp:503-505` writes `idx = (b_lo + b)*r` into `dst_blk_pos`, and `:597` in the general path. This is exactly the reference definition (tech report, `E999-qwen4exp-tech-report.md:117-121`: "Denoting the starting position of block b by p_b = b*r ... key compression is performed before positional encoding ... each compressed key is assigned the starting position p_b of its block").
* The norm weight is a constant.

So the answer to question 3 is unambiguous: **only the top_k selection is query-dependent. The gather, the four rect copies, the pooling adds, the scale, `index_k_norm` and the pooled rope are recomputing an immutable array of 32768 x 128 values, 12 times per token, when at decode exactly one block out of 32768 becomes newly complete every 4 tokens.** The redundancy factor at 131k is about 32768 x 4 = 131072x per new block, times 4 again because the indexer cache is mirrored so all four devices do it identically (`src/llama-model.cpp:510-513`: "the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split", `GGML_BACKEND_SPLIT_AXIS_MIRRORED`; confirmed by E038's load log).

Two specific corrections this forces on the existing records:

* **E037 is wrong** that "rms_norm and rope stay, since they are position-dependent and re-applied per step" (E037, "What each planned fix can remove", H9 bullet). For *block* keys the position is the block start, which never moves. H9's value is therefore the full 11.1 ms, not the ~3.4 s of 14.8 s E037 estimated. E038 repeats the same underestimate ("H9 is the decode item, ~25 s at 131k, of which the gather, rope, norm and the strided slices are the parts that vanish" - that one is actually right, so E038 corrected E037 implicitly; E042 then lost it again).
* **E042 is wrong** that "H9 (skip-and-reuse across steps) ... changes output". E042 is conflating two different ideas. The backlog's H9 (`plans/backlog.md:90-96`) is "cache pooled indexer block keys", which is arithmetic-identical. The output-changing idea is the tech report's MTP trick of reusing the *selection* across speculative steps (`E999-qwen4exp-tech-report.md:163,185`). Pooling at write time is bit-reproducible if the accumulation order is kept; the backlog itself already says so ("the reference does not re-pool per step either - it pools at write time and stores block-level keys").

Also worth noting: this is not a qwen4exp-specific blunder. `src/models/deepseek4.cpp:471-517` (`build_hca_compressed_kv_from_state`) does the same thing for DeepSeek-V4 - `get_rows` over `DSV4_HCA_RATIO*n_blocks`, weighted reduce, `build_norm`, rope, per step. So the pattern is inherited from the in-tree lightning-indexer precedent, which is why nobody has fixed it, and why a fix here is arguably upstreamable.

### F5. The user's hunch (a) is half right, and the half that is wrong matters

"a per-layer fast cache of ~4096 most-recent cells; if the graph re-gathers and re-ropes everything per token instead of incrementally, that is the missing piece."

* There is no 4096-cell recent-window fast cache in this design. The always-included set is only the tail of the query's own incomplete block: `r = 4` cells (`qwen4exp.cpp:700-715`, `tail_cells` is `I32 [r, n_tps, n_stream]` at `:610`; tech report `:133,157`: "up to 512 complete blocks ... plus the tail tokens in the final incomplete block", K=2048, r=4). The budget is 512 blocks x 4 + 4 = 2052, matching the "2052-wide" FA in the trace.
* "re-gathers and re-ropes everything per token" - **true for the indexer block keys** (F3/F4), **false for the main attention K/V**. K is roped before the cache write (`:891` -> `:785`), V is written raw (`:786`), and neither is re-gathered: the FA kernel reads cache rows directly through the compaction indices.
* The cache the user is intuiting does exist in spirit, but it is a cache of **pooled block keys** (1/4 the size of the raw cache), not of recent cells. That is F4 and it is the fix.

### F6. `copyBufferRectAligned` caller identified exactly: `ggml_cont` of a strided view, taking ggml-cuda's memcpy2D fast path

Not `stretch_2d` (no such symbol exists in `ggml/src/ggml-cuda/`), not a non-contiguous `cpy` kernel. The chain is:

1. `qwen4exp.cpp:643-647`: `slice = ggml_cont(ggml_view_3d(members, idx_dim, n_blocks, n_stream, members->nb[2], members->nb[3], i*members->nb[1]))`, inside `for (i = 0; i < r; ++i)` with `r = 4`.
2. `ggml_cont` -> `GGML_OP_CONT` (`ggml/src/ggml.c:3670-3686`).
3. `ggml-cuda.cu:2170-2171` -> `ggml_cuda_dup` -> `cpy.cu:622-625` -> `ggml_cuda_cpy(ctx, src0 = strided view, dst)`.
4. `cpy.cu:474-478`: `ggml_cuda_cpy_as_memcpy_2d(...)` succeeds (same type F32, same shape, contiguous prefix of 512 B at d=1, `ne[2] == 1`, `spitch = 2048 >= width`) -> `cudaMemcpy2DAsync` -> `hipMemcpy2DAsync` (`vendors/hip.h:93`) -> `__amd_rocclr_copyBufferRectAligned`.

The count is a perfect match: 130948 - 56980 = 73968 extra calls = **48.03 per token per GPU = 4 per QSA layer = `r`**. Each moves 16.8 MB out of the 67 MB f32 `members` at a 2048 B source pitch: 33.6 MB / 61.5 us = 546 GB/s, i.e. the copies themselves are efficient. They exist only because `ggml_add` needs contiguous rows; the comment at `:641` ("r is small, so summing slices beats a transpose plus sum_rows") was a reasonable choice before anyone measured it at 131k.

Note the same fast path is gated on `n_stream == 1` (`cpy.cu:417-421` requires `ne[i] == 1` above the row dim), so at batch>1 these become scalar `cpy` kernels instead - the concurrency behaviour of this chain will change shape under batching.

### F7. Two launch-config defects worth ~2.2 ms/token/GPU, both generic and both small

**(a) `rope_multi` wastes 75% of its threads on this shape and runs at 163 GB/s.** `rope.cu:472-476`:

```
const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
const int  n_blocks_x = (ne00 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
const dim3 block_nums(nr, n_blocks_x, 1);
```

For the pooled keys, `ne00 = idx_dim = 128` (`gguf-dump.log:39`) so `n_blocks_x = 1`, and in the kernel `i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y)` (`rope.cu:224`) with an early `return` when `i0 >= ne00` (`:226-228`): only 64 of 256 threads do work. One workgroup per block-row means 32768 workgroups of 8 wavefronts each moving 1 KB, of which 2 wavefronts are useful. Measured: ~209 us for 33.5 MB = **163 GB/s**, against 402 GB/s for the gather and 546 GB/s for the rect copies on the same arrays. Additional per-thread cost: `powf(theta_scale, iw/2.0f)` per element (`rope.cu:265-278`) and four divergent `pos[]` bases 128 KB apart (`blk_pos` is `I32 [4*n_blocks*n_stream]`, `qwen4exp.cpp:605`). Ceiling if it ran at its neighbours' rate: ~1.7-1.9 ms/token/GPU.

**(b) `flash_attn_mask_to_sparse_indices` launches ONE workgroup of 256 threads at decode.** `fattn.cu:110-111`: `blocks_num(mask->ne[1], mask->ne[3], 1)`, `block_dim(256,1,1)`. At decode the mask is `[131072, 1, 1, 1]`, so the grid is (1,1,1) and a single workgroup loops `131072/(256*8) = 64` times with 8 ballots, 3 `__syncthreads()` and a serial 8-warp prefix per iteration (`fattn.cu:44-90`). Measured 95.71 us/call = **0.57 ms/token/GPU for 262 KB of mask**, i.e. 2.7 GB/s. The kernel is an upstream NVIDIA design whose grid assumes many query rows (prefill); decode is its degenerate case. The index *order* is not semantically required by FA (it is a set), so a chunked multi-block version with a deterministic two-pass prefix, or simply a 1024-thread block, recovers most of it.

### F8. Unresolved data anomaly: FA-family kernels are exactly half the QSA layer count

12 QSA layers is established five ways in F0/F3. But `flash_attn_rtile<...,true>` = 9216, `flash_attn_mask_to_sparse_indices` = 9216, `flash_attn_combine_results` = 9240, `mul_mat_vec_q<(ggml_type)13,...>` = 9240 and `__amd_rocclr_fillBufferUnAligned` = 9240, all = **6.0 per token per GPU**, in *both* arms (no-qsa `flash_attn_rtile<...,false>` = 9240 too). Expected 12. The ratio is invariant to the divisor, so no re-normalization fixes it.

Two candidate readings, and they need separating before anyone quotes the FA rows:

* the FA rows are recorded for 2 of the 4 agents (9216 = 12 layers x 2 agents x 384 tokens), or
* FA is genuinely issued 24 times per token across the box = **2 per QSA layer = `head_count_kv`**, which would mean the attention is parallelized 2-way by KV head rather than 4-way by query head. That reading is self-consistent with cost: 337.48 us x 134 MB = 398 GB/s, and 24 x 134 MB = 3.2 GB = the entire 12-layer 131k KV footprint read exactly once per token across the box. But it is hard to reconcile with E038's finding that the per-device KV buffer is 1/4 of the total (`n_embd_k_gqa = 2 x 256 = 512`, split on axis 0 per `src/llama-model.cpp:537-539`), since 512/4 = 128 is half a head and FA needs a whole head.

Either way it does not change the priorities (0.64 ms/token/GPU becomes at most 1.28, still 10x below the indexer re-derivation), but E042's "sparse FA is negligible (0.07 ms/token/GPU)" and "mask_to_sparse_indices+FA 0.64" should be treated as **lower bounds of up to 2x**. Cheap diagnostic: re-run `rocprofv3 --kernel-trace` and group the CSV by `Agent_Id`/`Queue_Id`, counting `flash_attn_rtile` against `fill_kernel<__half>` (the 12-layer control) per agent.

### F9. One positive result E042 under-claims: the sparse FA port now pays at decode

Within the qsa arm, sparse attention costs compaction 0.57 + rtile 0.07 = 0.64 ms/token/GPU; the same 12 layers dense cost 2.03 (no-qsa `flash_attn_rtile<...,false>`, 337.48 us/call). So `Q4EXP_SPARSE_FA=1` is worth ~1.4 ms/token/GPU at decode, and 28x per call. That contradicts the standing note in `plans/backlog.md:56` (H4b: "tg flat exactly as P4 predicted"), which was recorded before the rtile + block-selection work landed. Worth a record update.

### F10. Minor: E042's `k_bin_bcast` rows are not comparable as quoted

E042 quotes "1.69 vs 0.09" from single rows. The qsa arm has three add rows (277200/2594.63, 18480/53.80 f16, 6160/9.87) and no-qsa two (215600/282.34, 73920/103.94); the truncated names hide template args, and the call-count deltas do not line up with the node count (qsa has 6160 *fewer* f32 add calls than no-qsa while having 84 more add nodes per token per GPU). The defensible number is the family total: 2658.30 vs 386.28 ms = **+1.48 ms/token/GPU**, not +1.60. Same caution applies to any per-row `k_bin_bcast` claim.

---

## PART 2 - PER-KERNEL MAPPING (question 2)

All line refs are `src/models/qwen4exp.cpp` unless stated. Shapes are decode, n_kv = 131072, r = 4, n_blocks = 32768, idx_dim = 128, n_idx_h = 4, n_stream = 1, budget 2052.

### `k_get_rows_float<__half, float>` - 18480 calls, 4596.69 ms, 248.74 us, 2.99 ms/token/GPU

* **Producer:** `:638` `members = ggml_get_rows(k_all, inp->blk_cells)`.
* **Reads:** `k_all` = the **indexer** KV cache view `[128, 131072, 1]` f16 (`:634-635`; cache created at `src/llama-memory-hybrid-idx.cpp:57-66`, type from `-ctk`, f16 by default, no V allocated, `rope_type = LLAMA_ROPE_TYPE_NONE` at `:54-56`).
* **Indices:** `blk_cells` `I32 [r*n_blocks, n_stream] = [131072, 1]` (`:604`), uploaded from host each ubatch by `set_input_qsa` (`src/llama-memory-hybrid-idx.cpp:484` fast path, `:620` general path).
* **Writes:** `[128, 131072, 1]` f32 = 67 MB (reshaped to `[128, 4, 32768, 1]` at `:639`). `ggml_get_rows` always emits F32, which is why an f16 cache produces an f16->f32 kernel and why everything downstream runs at 2x the bytes. E037's "one row the aggregates cannot name" is this row, and its H15 (f16 staging) guess was right about the dtype but wrong about the node.
* **Not** the main K/V cache, and **not** 2048 rows. The only other get_rows in the QSA graph are `:708` (`k_get_rows_float_vec<int>`, 18432 calls, 40.70 ms - the block->cell expand) and `:718` (cell-selection path, unused when `block_sel`).

### `__amd_rocclr_copyBufferRectAligned` - 130948 vs 56980, +73968 calls, +4547 ms, 2.95 ms/token/GPU

* **Producer:** the four `ggml_cont(ggml_view_3d(members, ...))` at `:643-647`. Path: `GGML_OP_CONT` (`ggml.c:3670-3686`) -> `ggml-cuda.cu:2170-2171` -> `cpy.cu:622-625` -> `cpy.cu:474-478` `cudaMemcpy2DAsync` (eligibility test `cpy.cu:391-431`).
* **Geometry:** width 512 B (128 f32), height 32768, spitch 2048 (`members->nb[2]`), dpitch 512. 4 per QSA layer, 48.03/token/GPU measured.
* **Baseline:** the 56980 no-qsa calls (1.32 us each) are the GDN conv `ggml_concat` path (`:1220`), which is also why `concat_cont` is 56980 in that arm. The qsa arm's extra 18480 `concat_cont` calls are `:715` `ggml_concat(cells, tail_cells, 0)`.

### `rope_multi<true,false,float>` - 55440 vs 18480, +36960 calls, +3886 ms, 2.52 ms/token/GPU

Three distinct producers, all `ggml_rope_multi` with `sections` (mrope, mode bit 8):

1. `:658` pooled block keys, `[128, 1, 32768]`, positions `blk_pos` = block start (`llama-memory-hybrid-idx.cpp:503-505`, `:597`). **This is the expensive one** (~209 us/call, ~2.50 ms/token/GPU).
2. `:667` indexer query, `[128, 4, 1]`, positions `inp_pos`. Tiny (~2 us).
3. `:885` and `:891` main Q and K, per full-attn layer. Tiny at decode (1 token); this is the 18480 no-qsa baseline (12/token/GPU = 1 per layer, i.e. Q and K appear as one row because they are the same instantiation).

**K is stored already-roped.** `:891` ropes `Kcur`, then `build_attn_qsa` writes it with `cpy_k` at `:785`. There is no "stored un-roped, re-roped per token" pattern on the main attention path. The pattern exists only on the indexer path, by explicit design (`llama-memory-hybrid-idx.cpp:54-56`), and that design decision is what F4 says is cacheable.

### `k_bin_bcast` add - +1.48 ms/token/GPU (family total)

* `:647` three `[128, 32768]` f32 pooling adds per layer - the bulk of the time (36/token/GPU).
* `:687` three head-sum adds over `[32768, 1, 1]` (n_idx_h = 4) - small.
* `:692` `score = ggml_add(score, inp->bias)` over `[32768, 1, 1]` - small; `bias` is the per-**block** visibility mask uploaded from host (`llama-memory-hybrid-idx.cpp:676-682`), which is what H13 bought.
* `:815` `kq_mask_top_k = ggml_add(kq_mask_top_k, kq_mask)` - the f16 row, 18480 calls, 53.80 ms, 12/token/GPU, exactly 1 per QSA layer.
* Not a bias add and not a concat. The mask add is the only f16 one.

### `rms_norm_f32<256, true, false>` - 110880 vs 73920, +36960 calls, +1571 ms, 1.02 ms/token/GPU

Template is `<block_size, do_multiply, do_add>` (`ggml/src/ggml-cuda/norm.cu:76`), so `true,false` = **the rms_norm+mul fusion is already firing** (`ggml-cuda.cu:4222-4225`, `norm.cu:366`); `build_norm` emits RMS_NORM then MUL adjacently (`src/llama-graph.cpp:1627,1641`).

The extra 24 calls/token/GPU = 2 per QSA layer:

* `:654` `build_norm(pooled, index_k_norm, ...)` over `[128, 32768]` - **carries essentially all of the 1571 ms** (~85 us/call, 33.5 MB per call = 394 GB/s).
* `:666` `build_norm(q, index_q_norm, ...)` over `[128, 4, 1]` - ~1.5 us, negligible.

The 48/token/GPU baseline present in both arms is one fused norm per layer elsewhere (attn_k_norm / ssm_norm), and `rms_norm_f32<256,false,false>` (110880 in both arms, 1.38 us) is the unfused variant. E042's "find the extra ~96/token" is answered: 96 per token across the box = 24 per GPU = `index_k_norm` + `index_q_norm`, and only `index_k_norm` costs anything.

### `flash_attn_mask_to_sparse_indices` + `flash_attn_rtile<...,true>` - 9216 each, 882.08 + 110.98 ms

* Producers: `fattn-common.cuh:1095-1103` (one compaction per FA op when `use_sparse`), `fattn.cu:102-118` (grid), `fattn.cu:20-100` (kernel). Enabled from the model side by `:825` `n_kv_max = use_sparse_fa ? top_k->ne[0] : 0` -> `build_attn_mha` at `:827` -> `src/llama-graph.cpp:2638`. Gate: `fattn.cu:126-141`, needs `K->ne[1] >= max(4096, 2*n_kv_max)` = 4104, satisfied at 131k.
* Input: the f16 mask built at `:792-815`. **The compaction re-derives, by scanning 131072 mask entries, the 2052 cell ids the graph already had in `top_k` at `:715`.** That round trip (indices -> fill+set_rows+add -> scan -> indices) is the 0.57 + 0.10 ms at the bottom of the F3 table.
* Count anomaly: see F8.

### `top_k_radix_*` - 851.59 ms total, 0.55 ms/token/GPU

`:705` `ggml_top_k(score, 512)` over `[32768, 1, 1]` f32. HIP radix path (`ggml/src/ggml-cuda/top-k.cu:186-210`): init + 4x(histogram, select) + reset + gather = 11 launches, counts 18432/73728/73728/18432/18432 all matching 12 top_k calls per token per GPU. H13 did its job: the input is 32768 blocks, not 131072 cells, and `select` is 8.73 us.

### `mul_mat_vec_f<float,float,4,64>` - 18480 calls, 905.44 ms, 49.00 us, 0.59 ms/token/GPU

`:674` `ggml_mul_mat(pooled, reshape(q, idx_dim, n_idx_h*n_tps, n_stream))`. Reads the 16.8 MB pooled array per layer per token: 343 GB/s. **This is the only per-token indexer cost that survives a pooled-key cache** (and it would halve if the pooled cache were f16).

---

## PART 3 - HOW LLAMA.CPP FUSES OPS TODAY, AND WHETHER THAT METHOD IS A QUICK WIN HERE (question 4)

### The in-tree mechanisms

1. **FA absorbs mask, scale and softmax.** `src/llama-graph.cpp:2670` registers `LLM_FUSED_OP_FLASH_ATTN`; the mask is a graph input (`:1013`, F16 when `cparams.flash_attn`), the scale is an op param, and softmax/online-max live inside `fattn-*.cu`. This is the "one kernel swallows five scalar ops" pattern the user is thinking of, and the QSA path already benefits from it (`:827`).
2. **A generic CUDA-side subgraph fusion matcher.** `ggml_cuda_try_fuse` (`ggml-cuda.cu:3503`) walks the cgraph and tries op-list patterns via `ggml_cuda_can_fuse` (`:3251`) / `ggml_can_fuse_subgraph_ext` (`ggml/src/ggml.c:7760`), which require **exact op match at consecutive node indices**, the COMPUTE flag, no external consumer of intermediates, and not an OUTPUT. Live patterns:
   * `{RMS_NORM, MUL, ROPE, VIEW, SET_ROWS}` and `{RMS_NORM, MUL, ROPE}` -> `rms_norm_mul_rope_f32` (`ggml-cuda.cu:3297-3326`, dispatch `:4206-4213`, kernel `rope.cu:711-850`).
   * `{ROPE, VIEW, SET_ROWS}` -> rope writes straight into the KV cache (`ggml-cuda.cu:2737-2769`, `:3327-3338`; `rope.cu:551-556`, `:704-706`).
   * `{RMS_NORM, MUL}` and `{RMS_NORM, MUL, ADD}` -> `rms_norm_f32<...,true,...>` (`:4216-4225`). **This one is firing on qwen4exp today** (F3/rms_norm row above), and commit `41abbfd59` is what enabled it.
   * `{MUL_MAT, ADD, MUL_MAT, ADD, GLU}` and the `MUL_MAT_ID` variants -> fused FFN gate/up (`:3264-3295`, predicate `:1744`).
   * `{SSM_CONV, ADD, UNARY(SILU)}` (`:3401-3420`).
   * `moe_weighted_reduction` (`:3514-3523`) and the whole `topk_moe` chain - sigmoid/sqrt-softplus/softmax + reshape + argsort + view + get_rows + sum_rows + clamp + div + scale collapsed into `topk_moe_cuda` (`:3538-3600`). That is the most aggressive compound fusion in the tree and it is the closest structural analogue to what the QSA chain needs.
   * `mul_mat_vec` + bias (`:4150-4200`, predicates `:1838`, `:1865`).
   * `gated_delta_net` + its strided state `cpy` (`:2824-2885`, `:3525-3536`) - precedent for a model-specific fusion that *elides a cache write*.

### Why none of the useful ones reach the QSA indexer chain

Two independent blockers, both citable:

* **The rope-mode gate.** `ggml_cuda_should_fuse_rope_set_rows` (`ggml-cuda.cu:2763-2767`) and `ggml_cuda_should_fuse_rms_norm_mul_rope` (`:2803-2807`) both `return false` unless `mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX`. qwen4exp uses mrope: `sections = {11,11,10,0}` (`gguf-dump.log:20`), and the observed kernel is `rope_multi`, which `rope.cu:660-671` only reaches when `mode & GGML_ROPE_TYPE_MROPE` and not `mode & GGML_ROPE_TYPE_NEOX` (`ggml.h:250-254`: NEOX=2, MROPE=8). Corroborating detail: `rope_multi_cuda` (`rope.cu:446-490`) has **no** `row_indices`/`set_rows_stride` parameters at all, unlike `rope_norm_cuda`/`rope_neox_cuda` (`:357-400`, `:401-445`) - the mrope path was never wired for the cache-write fusion.
* **Node adjacency.** Even ignoring the mode gate, the indexer's norm and rope are separated by a RESHAPE: `:653` reshape -> `:654` `build_norm` (RMS_NORM, MUL) -> **`:657` reshape** -> `:658` ROPE -> `:661` reshape. `ggml_can_fuse_subgraph_ext` matches by consecutive node index (`ggml.c:7766-7778`), so `{RMS_NORM, MUL, ROPE}` cannot match. The same reason the main K's `{RMS_NORM, MUL, ROPE, VIEW, SET_ROWS}` 5-node pattern misses: `:872` norm and `:891` rope are separated by the gate view (`:876-879`), the V reshape (`:882`) and the Q rope (`:885`).

So the honest answer to the user's hunch (b) is: **the method is exactly right and the tree already has the machinery, but on this graph the machinery is switched off by a rope-mode check and by node ordering.** A `{gather, 4x cont, 3x add, scale, rms_norm, mul, rope}` compound cannot be expressed as a pattern match at all - it is 10 nodes with a fan-out (four `cont` siblings all consuming `members`), which `ggml_can_fuse_subgraph_ext`'s single-consumer rule (`ggml.c:7792-7800`) is not shaped for.

### Verdict on "is a compound kernel a realistically quick win?"

**Yes for a hand-written kernel; no for the fusion-matcher route.** Concretely:

* The *matcher* route would need a 10-node pattern with sibling fan-out, which means extending `ggml_can_fuse_subgraph_ext` semantics. That is upstream-hostile and brittle. Do not plan on it.
* The *bespoke kernel* route is bounded and has direct in-tree precedent (`topk_moe`, `moe_weighted_reduction`, `gated_delta_net_fused_cache`, `rms_norm_mul_rope_f32`). Inputs it needs, all already live at that point in the graph:
  * `blk_cells` `I32 [r*n_blocks, n_stream]` (graph input, `:604`, host-filled at `llama-memory-hybrid-idx.cpp:484`/`:620`) - or, in the contiguous fast path, nothing at all, because the mapping is affine: `cur_blk_cells[(b-b_lo)*r + slot] = j` with `j = m + (j0 + b_lo*r - p0)` (`:466-495`), so `members` is a plain offset view of the cache.
  * the indexer cache K tensor (`:634`, f16, `[128, n_kv, n_stream]`),
  * `index_k_norm` weight `[128]` f32,
  * `blk_pos` `I32 [4*n_blocks*n_stream]` (`:605`, host-filled at `:503-505`/`:597`),
  * rope constants (`n_rot = 64`, `sections`, `freq_base`, yarn corr dims) - all op params,
  * output: `[128, n_blocks, n_stream]` f32 (or f16, which would also halve the `:674` matvec).
* Where the complexity lands: (i) one block-wide reduction over 128 elements for the rms-norm inside a kernel that is otherwise elementwise - `norm.cu:76-155` is the template to copy; (ii) the mrope section/theta logic - `rope.cu:250-292` is the template; (iii) a launch config that does not repeat F7(a)'s mistake, i.e. several block-rows per workgroup; (iv) bit-exactness - the mean must accumulate in the same order ((s0+s1)+s2)+s3 then *0.25, and the norm reduction order will differ from `rms_norm_f32`'s, so expect last-bit differences and validate with PPL plus a selection-overlap count, not equality (the same acceptance-test lesson E037 recorded for H13).
* Expected size: ~200-300 lines of HIP in one new `.cu`, plus ~20 lines of graph change to emit the fused node (or to make the pattern recognizable), plus a `Q4EXP_*` env gate matching the existing style (`:592` `Q4EXP_CELL_SEL`, `:824` `Q4EXP_SPARSE_FA`).

---

## PART 4 - RECOMMENDATION LIST

### 1. First: pool the indexer block keys at write time into a persistent per-layer cache (backlog H9). Ceiling ~11.1 ms/token/GPU of 31.9 (35%), plus a quadratic-to-linear prefill fix

This is the single most promising change and the user's hunch (a) is pointing at it, just one cache over. Design:

* New cache `[idx_dim, n_blocks, n_stream]` (f16 or f32) alongside the raw indexer cache, `GGML_BACKEND_SPLIT_AXIS_MIRRORED` like `cache_idx_(k|v)` (`src/llama-model.cpp:510-513`). 4x smaller than the raw cache it summarizes (33.5 MB -> 8.4 MB f16 per layer at 131k).
* Per ubatch, write only the newly completed blocks. The host already knows exactly which they are: `bid_idx`/`bid_cell`/`n_bid` (`llama-memory-hybrid-idx.cpp:498-506` fast path, `:565-575` general path), and there is already a "spare block" padding idiom for fixed-shape writes (`:598-600`). At decode that is 0 or 1 block per 4 tokens per layer.
* The graph then reads the pooled cache directly for `:674` and deletes `:638-661` entirely (gather, 4 conts, 3 adds, scale, norm, rope) on the fast path.
* Invalidation is the real cost, and the backlog already scoped it honestly (`plans/backlog.md:90-96`): block membership is positional, so `seq_add`/`seq_div` (context shift), `seq_rm`/`seq_cp`/`seq_keep`, defrag, clear, and state save/load (`llama-memory-hybrid-idx.cpp:211-250`) all have to be handled. Recommended shape: a "dirty from block B" watermark for appends, plus a **full-recompute fallback that is literally today's code path**, selected per graph build the way `block_sel` already is (`qwen4exp.cpp:597-615`, with `can_reuse` at `:505-537` gating graph reuse). That keeps the risky surface to "did we notice the mutation", and a missed invalidation is detectable by a selection-overlap test rather than silent.
* Bonus, not in the current records: this also removes most of the host-side `set_input_qsa` work, which is O(n_kv) per ubatch by its own admission (`llama-memory-hybrid-idx.cpp:312`, "~865 us at 33k", so ~3.4 ms at 131k, plus ~1.2 MB of H2D per token for `blk_cells`/`blk_pos`/`bias`). `blk_cells` stops being needed at all, and `blk_pos` only for new blocks. That is backlog E028 solved as a side effect, and it lands in the ~11 ms of non-kernel wall that F1 shows is real.
* Prefill: today the pooling is O(n_kv) per ubatch, so 131k prefill pays O(n_kv^2). E038 measured the prefill part of the chain at 71 s of 430 s device time at 131k. Pooling at write time makes it O(n_kv). This is worth more than the decode saving and is the argument for doing 1 before 2.

### 2. Second, if 1 is deferred: one compound kernel for `:638-661` (gather + mean + scale + rms_norm + mul + rope). Ceiling ~9.5-10.5 ms/token/GPU, none of the memory-semantics risk

Traffic goes from ~503 MB per layer per token (100 gather + 134 conts + 151 adds + 34 scale + 34 norm + 34 rope + 17 matvec) to ~50 MB (33.5 read f16 + 16.8 write f32), i.e. 606 MB/token/GPU instead of ~5.8 GB. Output-preserving up to reduction order. Section 3 above lists the inputs and where the complexity lands. **This is materially smaller than the user believes** - it is one kernel plus a graph tweak, with four in-tree precedents, and it does not touch `llama-memory-hybrid-idx` at all. It does not fix prefill's quadratic behaviour or the host scan, which is the only reason to prefer 1.

### 3. Third and fourth: the two launch-config fixes from F7. ~2.2 ms/token/GPU combined, ~50 lines, generic

* `fattn.cu:110-111` - the decode compaction grid is (1,1,1) with 256 threads over 131072 mask entries. Chunk over `ne30` with a deterministic two-pass prefix, or at minimum widen the block. Ceiling 0.57 ms/token/GPU (1.15 if F8 resolves to 2x). Do this one even if 1 or 2 lands, since neither removes the compaction.
* `rope.cu:472-476` - `rope_multi` puts one row per workgroup with `block_dims(1, 256, 1)`, and at `ne00 = 128` only 64 of 256 threads survive the `i0 >= ne00` early-out (`:224-228`). Pack multiple rows per workgroup when `ne00 < 2*CUDA_ROPE_BLOCK_SIZE`. Ceiling ~1.7-1.9 ms/token/GPU on this model, and it helps every mrope model with a small head dim. Skip if 1 or 2 lands (both delete this rope call), otherwise it is the cheapest real win on the board.

### 4. Also worth doing, cheap, no perf motive

* `:803` allocates `zeros` as F32 while the in-tree precedent at `deepseek4.cpp:694` uses `cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32`. Cosmetic at decode (`fill_kernel<float>` 20 ms total) but it is a divergence from the pattern the file says it copies ("copies the MLA sparse path", `:753`).
* Records to correct: E042's totals and busy fractions (F1), the `k_get_rows_float<half>` / rope / copyBufferRect attributions (F2, F6), "H9 changes output" (F4); E037's "rms_norm and rope stay, they are position-dependent" (F4); `plans/backlog.md:56` H4b's "tg flat" (F9).

### 5. Explicitly not worth doing now

* Fusing the mask build away by passing `top_k` indices straight to FA. It saves 0.67 ms/token/GPU at decode and needs a `ggml_flash_attn_ext` API change; the prefill payoff is larger (the mask is `[n_kv, n_tps]` f16 per layer per ubatch) but so is the disruption. Park it behind 1.
* Anything about NCCL (out of scope per instruction) and anything about splitting the 4x-mirrored indexer chain across devices. The latter is real - F4 shows all four GPUs compute bit-identical indexer results because the cache and projections are mirrored (`src/llama-model.cpp:510-513`) - but splitting it trades device time for collectives, and 1 makes the point moot by deleting 84% of the work on every device at once.

### Residual risks / validation gaps

* F8 is unresolved and is the one number in E042 I would re-measure before quoting FA figures. Diagnostic given there.
* My per-stage split of the adds (F3, "1.48 for the family") is solid as a total but not per node; the truncated `k_bin_bcast` names do not allow attributing rows to `:647` vs `:687` vs `:692` vs `:815`. If the pooling adds need their own number, the CSV grouped by full demangled name gives it.
* The 163 GB/s rope figure (F7a) is inferred by subtracting the tiny `:667` calls from the family delta; it is ~209 us/call only if the 12 big and 12 small ropes per token per GPU split the 3886 ms as I assumed. A per-node timestamp from the raw CSV would confirm.
* Bandwidth ceilings assume ~500-640 GB/s effective. The adds appearing to run at >1 TB/s in the aggregate suggests RDNA4 Infinity Cache is absorbing the 16.8 MB pooled arrays, which would make the fusion saving in option 2 somewhat smaller than the pure-traffic model predicts (the gather's 67 MB write and the cache's 33.5 MB read cannot be cached away, so the floor on the saving is still large).
* Everything here is decode at 131k, single stream (`n_stream = 1`, proven by the memcpy2D eligibility gate at `cpy.cu:417-421`). At batch > 1 the rect copies become scalar `cpy` kernels and the mask grows by `n_tps`, so the ranking of 3 versus 1/2 may shift.

---

## Parent deposition note (not part of the review)

- The review's record corrections (F1, F2, F4, F9) were applied: E042 corrected inline, E037 H9 bullet
  fixed, backlog H9 re-prioritized with the measured prize and H4b "tg flat" superseded.
- The review's recommendation ordering (1 H9, 2 compound kernel, 3+4 launch configs) is the current
  plan; no code has been written for any of them.
