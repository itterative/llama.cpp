---
name: rdna4-rocm-build
description: How to build and measure llama.cpp HIP on RDNA4/gfx1201 - cmake paths, knobs that are inert, RDNA4 kernel behaviour, FA family selection, and the qwen4exp op-support verdicts.
category: project
priority: 4
keep_updated: true
---

# RDNA4 / HIP build + backend behaviour

Anchors at `ebbb18522`. Marked **[v]** = read/verified by me directly; **[s]** = from the
`scout-1` survey, plausible but not independently re-read; **[x]** = corrected after
checking. Line numbers drift; trust the symbol names.

## Build

HIP is a thin cmake wrapper that **compiles the CUDA sources** with the HIP toolchain:
`ggml/src/ggml-hip/CMakeLists.txt:58-73` globs `../ggml-cuda/*.cu|*.cuh` plus the
`fattn-*`/`mmq`/`mmf` template instances. `GGML_HIP` reaches it through `ggml_add_backend(HIP)`
(`ggml/src/CMakeLists.txt:593`, macro `:428-439`), which defines both `GGML_USE_CUDA` and
`GGML_USE_HIP`.

**Two mutually exclusive target paths** (`[v]` `ggml/src/ggml-hip/CMakeLists.txt:19-24`):

| `CMAKE_CXX_COMPILER` | effect |
|---|---|
| matches `hipcc$` | legacy: warns, no `enable_language(HIP)`, sources as `LANGUAGE CXX`, **`GPU_TARGETS`/`AMDGPU_TARGETS` silently ignored**, hipcc auto-targets the local GPU |
| anything else | `AMDGPU_TARGETS` -> `GPU_TARGETS` -> `CMAKE_HIP_ARCHITECTURES` (`:36-41`), `enable_language(HIP)`, `LANGUAGE HIP` |

**Our `build/` takes the second path** `[v]`: `CMAKE_CXX_COMPILER=/usr/lib64/ccache/c++`
(not hipcc) and `CMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++` is populated. So
`-DGPU_TARGETS=gfx1201` **does** work here, and its absence from the cache means the target
came from CMake's native "all GPUs in this system" detection, not from a hipcc fallback.
Pin it explicitly for anything that has to be reproducible:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
      -DGPU_TARGETS=gfx1201 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

Only one ROCm-version gate exists in cmake: **ROCm >= 6.1 required**
(`:54-56`) `[v]`. There is **no** `#if ROCM_VERSION >= ...` anywhere in `ggml/src/ggml-cuda/`
`[s]`; in-code version sensitivity is via `HIP_VERSION` (`vendors/hip.h:163-175`, `:251-255`)
and it only affects fp8 + CDNA, so a 6.4 -> 7.x upgrade changes less here than feared -
but it is still a comparability break for measurements.

`GGML_STATIC` is a hard `FATAL_ERROR` on the HIP path (`:136-138`) `[s]`, which is why the
build is `BUILD_SHARED_LIBS=ON` and why the loader-shadowing trap on this box exists.
Workaround instead of static: set `CMAKE_INSTALL_RPATH`/`CMAKE_BUILD_RPATH` to `$PWD/build/bin`
(backlog B0).

## Knobs: what is real and what is inert on RDNA4

| knob | status on gfx1201 |
|---|---|
| `GGML_HIP_NO_VMM` default **ON** | real. `ggml-cuda.cu:278-293` then reports `vmm = 0` **without querying the device** - the `VMM: no` banner is this flag, not a hardware statement `[s]` |
| `GGML_HIP_GRAPHS` default ON | real; CUDA-graph capture is auto-disabled only for `cc < VOLTA`, so RDNA4 keeps graphs `[s]` |
| `GGML_HIP_MMQ_MFMA` default ON | **inert on RDNA4** `[v]`: `AMD_MFMA_AVAILABLE` requires `defined(CDNA)` (`common.cuh:274-276`); RDNA4 gets `AMD_WMMA_AVAILABLE` (`:279-281`) |
| `GGML_CUDA_GRAPHS`, `GGML_CUDA_NO_VMM`, `GGML_CUDA_COMPRESSION_MODE` | ignored on the HIP path; use the `GGML_HIP_*` equivalents `[s]` |
| `GGML_CUDA_DEBUG=ON` | **no-op on HIP** `[s]`: only `ggml/src/ggml-cuda/CMakeLists.txt` reads it, `ggml-hip/CMakeLists.txt` never does. Need an explicit define in the compile flags to get those logs |
| `GGML_CUDA_FA`, `GGML_CUDA_FA_QUANTS`, `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS`, `GGML_CUDA_NO_PEER_COPY` | honored on HIP via the shared `ggml/cmake/common.cmake:52-119` `[s]` |
| `GGML_HIP_RCCL` default OFF | and the "rebuild with NCCL" warning is `#ifndef GGML_USE_HIP` (`ggml-cuda.cu:1197-1201`) `[s]` - so without RCCL you get **no warning at all**, just silent fallback to internal AllReduce |

Runtime env that matters here (all `[s]` unless noted):

- `GGML_CUDA_DEVICES` - exposes N *virtual* devices round-robined over physical GPUs
  (`ggml-cuda.cu:235-259`) **[v] present**. **This is how multi-device `-sm layer` paths get
  tested on the 16 GB box.** Unblocks a class of T1 work I had assumed needed T2.
- `GGML_CUDA_DISABLE_FUSION` (`:3434`, `:4507`) - A/B fusions without rebuilding.
- `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_DISABLE_GRAPHS`, `LLAMA_GRAPH_REUSE_DISABLE`.
- `GGML_OP_OFFLOAD_MIN_BATCH` (default 32, `:5717`) - interacts with
  `get_op_batch_size()`, where `GET_ROWS -> 0` and `MUL_MAT -> ne[1]`.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` - spills to sysram rather than OOM-aborting; the
  mitigation for big `test-backend-ops` cases.
- `GGML_CUDA_P2P` (opt-in), `GGML_CUDA_ALLREDUCE={nccl,internal,none}`,
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE={f32,f16,bf16,auto}`, `GGML_CUDA_NO_PINNED`.
- Device names are **`ROCm0`, `ROCm1`, ...** `[v]` (`ggml/include/ggml-cuda.h:11`), and
  `test-backend-ops -b` is an exact `strcmp` (`tests/test-backend-ops.cpp:12014`) that
  **exits 0 having tested nothing** if you typo it `[v]`.

## What RDNA4 does differently

- **`should_use_mmq` returns true unconditionally on RDNA4** (`mmq.cu:380-382`; the in-source
  note says MMQ beats dequant+hipBLAS for every type/batch, ref PR #18537) `[s]`. So quantized
  matmuls are always MMQ here - `GGML_CUDA_FORCE_CUBLAS` experiments are a different ISA path.
- `prefer_f32_output` is **forced on** for RDNA4 (`ggml-cuda.cu:1512`, `:1514`) `[v]`.
- MMQ tile selection uses `ncols_opt = ceil(ne12*n_expert_used/ne02)` for MoE on RDNA3/4
  (`mmq.cu:248-251`) `[s]` - so with `num_experts_per_tok = 10` the tile choice is already
  expert-aware.
- Dedicated `mmq-config-rdna4.cuh` tile table; `MMVQ_PARAMETERS_RDNA4` plus an RDNA4-only
  `nwarps` whitelist (`mmvq.cu:417-418`, `:467-492`) `[s]`.
- `v_dot2_f32_f16` (`common.cuh:769-771`) and `__builtin_amdgcn_sudot4` for dp4a
  (`:724-731`); WMMA f16 16x16x16 w32 gfx12 in `mma.cuh:1024-1033` `[s]`.
- Device-side RDNA4 is switched on by `__GFX12__` -> `#define RDNA4`
  (`vendors/hip.h:215-217`) `[s]`; host-side by `gcnArchName` parsing ->
  `GGML_CUDA_CC_IS_RDNA4(cc)` true for gfx1201 (`common.cuh:84`, `:93`) `[s]`. **So host-side
  heuristics work even in a binary whose device code lacks the macro** - a subtle way to get a
  half-tuned build if `GPU_TARGETS` is ever set wrong.

## Flash attention on gfx1201, and what qwen4exp actually gets

FA is available (`FLASH_ATTN_AVAILABLE` is just `!GGML_CUDA_NO_FA`, `common.cuh:304-306`)
`[s]`. Families: `fattn-vec-*` (head 64/128/256 only), `fattn-tile-*` (generic fallback),
`fattn-mma-f16-*` (compiles under `AMD_WMMA_AVAILABLE`, i.e. RDNA3/RDNA4).

The WMMA rule is `fattn.cu:667-671` **[x]** - scout cited `:653-666`, which is the
`amd_mfma_available` branch and therefore **CDNA-only, never RDNA4**. The real one:

```c
if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
        Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
    return BEST_FATTN_KERNEL_MMA_F16;
}
```

plus `gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0 && K->ne[1] % 256 == 0`
(`:537`) `[v]`.

Derived for real qwen4exp dims (`head_dim 256`, `24 Q / 2 KV heads`, DK==DV==256 and
`(256,256,...)` instances exist `[v]`; `no_mtp`, and it asserts `n_embd_head_v ==
n_embd_head_k`): `gqa_ratio = 12` -> `gqa_ratio_eff = 4`, threshold for DK=256 is
`Q->ne[1]*4 > 16`, so:

- **prompt processing: mma_f16 FA.** Decode of a single token: **not** - it falls to
  tile/vec. That split matters because
- **sparse FA exists only in the mma_f16 family**, and on HIP it is doubly closed:
  `ggml_cuda_flash_attn_ext_compact_mask` -> `GGML_ABORT("sparse flash attention is only
  supported on NVIDIA CUDA")` (`fattn.cu:92-96`) and
  `..._mma_f16_shall_use_sparse` returns false (`:109-113`), with the `if constexpr`
  sparse branch itself `#if !defined(GGML_USE_HIP)` (`:133-140`) `[v]`.

So H4b's port surface is one warp-ballot kernel (`fattn.cu:10-89`, `WARP_SIZE == 32` which
gfx1201 satisfies, but its `ggml_cuda_pdl_*` hooks are NVIDIA-only) plus removing the
compile-time `#if !defined(GGML_USE_HIP)` guards (`:92-96`, `:109-113`, `:133-140`) `[v]`
plus the call site `src/models/qwen4exp.cpp:767`. Much narrower than "implement sparse
attention", but still kernel work.

**The `GGML_ABORT` is unreachable, and the call site cannot be flipped early.**
`ggml_cuda_flash_attn_ext_supported` -> `get_best_fattn_kernel != NONE` never consults
`n_kv_max`, so sparse-shaped cases report `SUPPORTED` on `ROCm0` - all 18 nonzero
`n_kv_max` cases in `test-backend-ops` do - and then **pass by computing dense over the same
mask**, because `use_sparse` is false and `compact_mask` is never called `[v]`. Consequence:
passing `top_k->ne[0]` today is **inert** on ROCm, identical results and identical cost. So
H4b cannot be staged by toggling that argument, and H4a cannot be probed that way either.

## qwen4exp op support on HIP - the headline is "everything is on the GPU"

Now backed by execution, not only by reading gates `[v]`: `test-backend-ops support -b ROCm0`
over 13 op families gives 9906 supported / 2384 unsupported cases, and `FLASH_ATTN_EXT` at
**hsk/hsv 256/256 is 142/142 supported** (f16, q4_0, q8_0 KV). Shape pairs with zero
support, so never assume them: 96/64, 128/64, 192/192, 64/128.

Against `ggml_backend_cuda_device_supports_op` (`ggml-cuda.cu:5064`), **every op in the
qwen4exp graph has a real HIP kernel**; the only `GGML_USE_HIP` conditionals in that function
*loosen* (`TOP_K :5457`) or tighten (`ARGSORT :5462`) rather than disable `[s]`. So P1's
premise - "some ops fall off the GPU" - is probably false, and the interesting list is much
shorter:

- **HC ops are F32-only, with no shape gate** `[v]` (`:5492-5501`). qwen4exp emits only
  `DSV4_HC_PRE` with `gated=1` (`src/models/qwen4exp.cpp:293`) and `DSV4_HC_POST` with
  `src[3] == nullptr` (`:338`); `_COMB` is DeepSeek-V4/Kimi-K3's, not this arch's `[x]`.
  Any f16 feeding those inputs silently leaves the fused path. The `ne[1] == 4` restriction
  I worried about (H3) is **Metal/Vulkan's**, not HIP's - H3 closed.
- `GATED_DELTA_NET` unconditional on non-MUSA (`:5485-5491`), with the ~40-node chunked
  fallback also fully covered `[s]`.
- `TOP_K` gets a dedicated radix-select kernel on HIP, unrestricted
  (`top-k.cu:51-211`, dispatch `:261-275`) `[s]` - the QSA budget cut is not the problem;
  the missing compaction is.
- `ARGSORT` (MoE router): without CUB - **`GGML_CUDA_USE_CUB` is never defined on HIP**
  `[v]` `common.cuh:113-115` - support requires `ne[0] <= 1024` `[v]`. Real
  `num_experts = 512`, so **it fits** and the router stays on the GPU. scout flagged this as
  the one possible mass-offload; the config closes it. Re-check if a future variant exceeds
  1024 experts.
- `SET_ROWS` dst whitelist is `{F32,F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,IQ4_NL}` `[s]` ->
  **a Q4_K/Q5_K/Q6_K KV cache is not supported for the KV writes**, which is a live
  configuration question since qwen4exp writes KV via `cpy_k`/`cpy_v` = `set_rows`.
- The 3-node `RMS_NORM+MUL+ROPE` fusion **rejects qwen4exp** because it only accepts
  `GGML_ROPE_TYPE_NORMAL`/`NEOX` and this arch is `IMROPE` (`ggml-cuda.cu:2734-2735`
  `[v]`, rope type from `llama-model.cpp:3063-3070`). A concrete, self-contained fusion gap.
- `SSM_CONV` needs `src[0]->ne[1] % 128 == 0` (`:5416-5418`) `[s]` - satisfiable but not
  structurally guaranteed; check against real GDN dims.
- `REPEAT` on CUDA is F32/F16-only (`:5307-5312`) `[s]`; all qwen4exp uses are F32, fine.
- `LIGHTNING_INDEXER` is **not in this graph** `[x]` - `build_qsa_top_k` hand-rolls the
  indexer from `MUL_MAT`/`ROPE`/`RELU`/`ADD`/`GET_ROWS`/`TOP_K` (`:542-691`).

## Traps

- **`docs/ops.md` and `docs/ops/*.csv` are stale and wrong for these ops** `[s]`: they mark
  `DSV4_HC_*` as unsupported on CUDA and Metal while the kernels exist, and `CUDA.csv` has no
  rows at all for `DSV4_HC*`/`GATED_DELTA_NET`/`LIGHTNING_INDEXER`/`TOPK`. Generated snapshot
  (`scripts/create_ops_docs.py`); never use it as support truth.
- `test-backend-ops` **works on gfx1201 / ROCm 6.4.4** `[v]`: 1500/1500 non-FA cases and
  3973/3979 FA cases passed, no hang. The reported AMD hang is therefore not universal;
  E001's update has the stages, and the bench-box hang stays open as a box-specific symptom.
  Two real sharp edges remain: there is **no insufficient-memory skip logic** `[s]` (an
  oversized case OOM-aborts), and `-b` is an exact `strcmp` that **exits 0 having tested
  nothing** on a typo `[v]`.
- known numeric defect, not on our path: `FLASH_ATTN_EXT` fails 6 cases at
  **hsk=192/hsv=128** (gqa 8/16, permuted K/V views), err up to 0.0298 vs a 0.0005 tol `[v]`.
  qwen4exp is 256/256, which passes 142/142. Know this before re-running the FA suite and
  seeing red.
- qwen4exp uses `llama_memory_hybrid_idx` (`llama-model.cpp:2565`, `:2584-2589`), **not** the
  `dsa`/`msa`/`iswa` cache classes; its `set_input_qsa` is host-side index/bias construction
  with `GGML_ASSERT(r <= 64)` (`llama-memory-hybrid-idx.cpp:273-340`) `[s]` - real
  `compress_ratio = 4`, fine.
- Sparse-ness in llama.cpp is expressed **as the mask**, not as a distinct kernel
  (`src/models/qwen4exp.cpp:735-758`); the compaction is what turns "masked" into "cheaper".

## Not yet verified

- Whether `__GFX12__` is actually emitted for `gfx1201` (nobody ran the compiler). If it is
  not, every device-side RDNA4 path above is dead code in this build while the host-side
  heuristics still claim RDNA4. **One `amd-smi`-free check settles it**: build one TU with
  `-D__GFX12__` probing, or grep the object for the gfx12 ISA.
- Whether `-cmoe`/`-ncmoe` and `-ot` placement interacts with `ARGSORT`/`MUL_MAT_ID` support
  gates when expert weights live on CPU.
