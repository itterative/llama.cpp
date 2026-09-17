---
name: qwen4exp-arch
description: What the qwen4exp arch (Qwen3.8-Flash-Next) actually is - code map, the three subsystems (hyper-connections, QSA, PLE), and the two facts that dominate its RDNA4 cost profile.
category: project
priority: 4
keep_updated: true
---

# qwen4exp (Qwen3.8-Flash-Next) architecture map

Arch string `qwen4exp` = `LLM_ARCH_QWEN4EXP` (`src/llama-arch.cpp:43`). HF id
`Qwen/Qwen3.8-Flash-Next`, registered as `Qwen4ExpForCausalLM` /
`Qwen4ExpForConditionalGeneration` (`conversion/__init__.py:243`).
Model impl `src/models/qwen4exp.cpp` (1298 lines, `llama_model_qwen4exp`). Converter
`conversion/qwen4exp.py`. Initial support: `6c84c7d5d`.

Per the converter docstring (`conversion/qwen4exp.py:19`): it is Qwen3.5's gated delta
net + interleaved mrope, plus three additions - **hyper-connections in place of every
layer norm**, **QSA sparse attention on the full-attention layers**, and **PLE n-gram
hash embeddings on a single layer**.

## Layer structure

Linear (GDN) attention everywhere except every `full_attention_interval`-th layer, which
is full attention with QSA (`src/models/qwen4exp.cpp:127`). No `output_norm`: the final
hyper-connection mixer *is* the output norm (`:159`, `:445`). MTP head is not exported
(`no_mtp`, `conversion/qwen4exp.py:22`).

## Hyper-connections (HC)

`hc_count` parallel residual streams shaped `[n_embd, hc, T]` replace per-layer norms
(`:265`). Low-rank mixers: `HC_ATTN_{NORM,DOWN,UP,INJECT}` and `HC_FFN_{...}` plus
`HC_HEAD_{NORM,DOWN,UP}` (`gguf-py/gguf/constants.py:820-827`, `:672-674`).
`hc_lowrank` is qwen4exp-specific - DeepSeek-V4 leaves it absent and runs full rank
(`src/models/qwen4exp.cpp:44`). `hc_count <= 1` is rejected at load, matching all three
reference configs (`:47`). Note the converter folds each gamma to `(1 + w)` (`:280`), and
`2*sigmoid` centres the scatter weights on 1 so a zero injection is a plain residual add
(`:331`) - both are load-time facts that a kernel reimplementation must reproduce.

Fused ops: qwen4exp emits only **`DSV4_HC_PRE` with `gated=1`** (`:293`) and **`DSV4_HC_POST`
with `src[3] == nullptr`** (`:338`) `[v]`. `_COMB` and the non-gated `_pre` belong to
DeepSeek-V4 / Kimi-K3, not this arch. Builders `ggml_dsv4_hc_comb/_pre/_pre_gated/_post` at
`ggml/include/ggml.h:2690-2722`; kernels landed in `37b53fd45`. On HIP the support gate is
dtype-only - **all inputs F32**, and no shape restriction (`ggml/src/ggml-cuda/ggml-cuda.cu:5492-5501`
`[v]`), unlike Metal/Vulkan which demand `hc == 4`. See `rdna4-rocm-build`.

**Cost profile: HC is per-layer and per-token, in every mode.** Nothing about it is
batch-size dependent, so it is a first-order target for both pp and tg.

## QSA - and the fact that dominates everything

Indexer + top-k block selection is built and its result **is** applied: the attention
mask is rewritten so that only the selected blocks survive. What is missing is the
*compaction*, so the kernel still iterates the full KV extent.

```cpp
// src/models/qwen4exp.cpp:764
// TODO: enable sparse attention when we are ready
// ref: https://github.com/ggml-org/llama.cpp/pull/27970
//ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
ggml_tensor *  cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0,               kq_scale, il);
```

The argument is `n_kv_max` (`src/llama-graph.h:1180-1190`) -> `ggml_flash_attn_ext_set_n_kv_max()`
(`src/llama-graph.cpp:2638`). Only `deepseek4.cpp:761` passes a nonzero value today.

What `n_kv_max` buys, from the CUDA side (`ggml/src/ggml-cuda/fattn-common.cuh:1095-1131`):

- `use_sparse`: compact the mask into a `KV_max` buffer of `n_kv_max * mask_rows` and set
  `n_kv = n_kv_max` - the kernel touches only the selected budget.
- otherwise: `n_kv = K->ne[1]`, i.e. **quadratic in full context length**. The only rescue
  is a tail-skip heuristic (`flash_attn_mask_to_KV_max`, `:1106-1122`) that skips trailing
  all-masked tiles; it does not reclaim interior sparsity, and it is gated on
  `K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)`.

So on qwen4exp today the indexer pipeline (`build_qsa_top_k`, `:542-691`: mul_mat, rope,
slice sums, rectified dot products, top_k) *and* the per-layer mask rebuild
(`ggml_fill(-INF)`, `ggml_set_rows`, `ggml_add`, `:735-758`, over
`[n_kv, n_batch, 1, n_stream]`) are paid for, while the saving they exist to unlock is not
collected. Semantics are right; the bill is what is wrong.

**Why it cannot simply be switched on: on ROCm, sparse FA aborts.**

```c
// ggml/src/ggml-cuda/fattn.cu:93
void ggml_cuda_flash_attn_ext_compact_mask(...) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
```

and `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse()` returns `false` under
`GGML_USE_HIP` (`fattn.cu:107`). The NVIDIA kernel to port is
`flash_attn_mask_to_sparse_indices` (`fattn.cu:10-32`): warp-ballot compaction,
`values_per_lane = 8`, assumes `WARP_SIZE == 32` - which RDNA4 satisfies (this box
reports `Wave Size: 32`). Its `ggml_cuda_pdl_sync()` / `ggml_cuda_pdl_lc()` are programmatic
dependent launch hooks and will need a non-PDL equivalent. Vulkan does have a shader
(`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_sparse_compact.comp`), so a second
reference implementation exists.

Consequence for planning: "make qwen4exp fast on RDNA4" is really two projects - (a) stop
paying for the indexer and mask rebuild without collecting the sparsity, (b) port the
compaction so the sparsity is collectable. (b) is the long-context pp win; (a) alone is a
smaller, safer win that is measurable at any context length. Doing neither means long
context costs the same as dense attention plus a tax.

## PLE - per-layer embeddings via n-gram hashing

`PLE_{KEY,VALUE,NORM_KEY,NORM_QUERY,NORM_CONV,CONV1D}`
(`gguf-py/gguf/constants.py:828-833`), on the layers named by `ple_layer_ids` (1-based in
the HF config, converted to 0-based in the exporter, `conversion/qwen4exp.py:70`).

Row selection is a hash: `mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1])`, then
`row = mixed_n % vocab[h] + offset[h]` (`src/models/qwen4exp.cpp:1045` block header).
**The hash is computed host-side**, because ggml has no int64 and no xor. That makes PLE a
per-ubatch CPU loop plus an H2D upload sitting in the graph input path, with the
predecessor tokens read out of the attention KV cells (`ext.tok`), and EOS resetting the
window. The int64 multipliers are read straight from the checkpoint to dodge a float32
rounding of 45-bit values (`conversion/qwen4exp.py:36-48`) - so any "just store them as
f32" idea is already known-broken.

The conv is depthwise causal, **dilated by the n-gram size**, written as a sum of shifted
copies rather than `ggml_conv_1d_dw`, because that op "is documented as unreliable"
(`src/models/qwen4exp.cpp`, PLE conv comment). Conv history lives in its own recurrent
cache rows - two per layer, which is why the shared `build_conv_state` is not usable.

**Under `-sm tensor` the PLE path is mirrored, not split**
(`GGML_BACKEND_SPLIT_AXIS_MIRRORED` for the PLE cache, `src/llama-model.cpp:513-515`: "the PLE
table is model-level and its conv is mirrored, so every device runs the whole conv and needs
the whole history"). Consequence for memory sizing: putting the ~30 GB table on GPU costs
~30 GB **per card**, and the QSA indexer cache is mirrored for the same reason (`:508-510`).
This is why decode-side work targets a row cache rather than the table - see
`experiments/plans/ple-prefetch.md`.

## Graph and memory

Always builds `llama_memory_hybrid_idx` (`:368`), so the recurrent+KV hybrid path is in
play; the indexer cache inside it is absent when the GGUF has no indexer tensors.
Recurrent-state rollback is supported for this arch (`llm_arch_supports_rs_rollback`,
`src/llama-arch.cpp:1112`) and needs the `[TAG_RECURRENT_ROLLBACK_SPLITS]` conv rows.
Graph-split reduction was already worked on in `6fe749801` and nodes are deliberately
co-added to prevent reordering (`:696` comment).

## Known gap specific to multi-GPU

`llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1130`) returns **false** for
`LLM_ARCH_QWEN4EXP` at `:1161`, tagged upstream `// TODO: fix test-llama-archs`. So
`-sm tensor` is unavailable; only layer split works. On 4 cards with heterogeneous layer
types this is the first structural limitation to hit.
