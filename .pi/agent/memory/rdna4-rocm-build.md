---
name: rdna4-rocm-build
description: How to build and measure llama.cpp HIP on RDNA4/gfx1201 - cmake paths, knobs that are inert, which matmul kernel MoE decode actually picks (mmvq vs mmq), RDNA4 kernel behaviour, FA family selection, and the qwen4exp op-support verdicts.
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
and it only affects fp8 + CDNA, so the 6.4 -> 7.1 upgrade this box actually made on
2026-09-17 changed nothing in the source - but it broke the **build tree**: the runtime SONAMEs
all moved (`libamdhip64` .6 -> .7, `librocblas` .4 -> .5, `libhipblas` .2 -> .3, all under
`/lib64`, and `/opt/rocm*` is gone entirely) so a tree built against 6.4 will not load `[v]`.

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

- **`should_use_mmq` is NOT what runs at decode - correcting an earlier version of this file.** It does
  return true for RDNA4 with `n_experts > 0` (`mmq.cu:380-386`, in-source ref PR #18537), but for MoE it is
  never consulted at batch 1: `ggml_cuda_mul_mat_id` returns into mmvq first when
  `ne2 <= get_mmvq_mmid_max_batch(type, cc)` (`ggml-cuda.cu:1993-2001`), and the same ordering applies to
  plain MUL_MAT (`:1936-1944`). See "MoE matmul dispatch" below. `[x]` the previous bullet here claimed
  "quantized matmuls are always MMQ here"; it was marked `[s]` and E053's decode trace disproves it (zero
  `mul_mat_q` rows in 1736 steps).
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

## MoE matmul dispatch at decode, and why the split axis matters (`[v]` from source, 2026-09-24)

**Decode MoE is mmvq, and it is the *tuned* branch.** The gate is a per-type batch cap,
`get_mmvq_mmid_max_batch_rdna4` (`mmvq.cu:258-282`): Q4_K 4, Q5_K 5, Q6_K 5, Q3_K 4, IQ2/IQ3 4,
Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_* 7, default 8. Batch 1 is below every one of them, so all of this model's
expert types go to mmvq and `should_use_mmq`'s `n_experts > 0` clause only ever fires on prefill. Inside
mmvq, `MMVQ_PARAMETERS_RDNA4` (`calc_nwarps`, `mmvq.cu:465-489`) gives **8 warps at `ncols_dst == 1`** for
exactly those types - so the measured ~115 GB/s is happening on the branch RDNA4 tuning was written to
favor, which is the point of the whole investigation.

**Two tables matter, not one:**
- `get_mmvq_mmid_max_batch_rdna4` (`mmvq.cu:258`) - who gets mmvq at which batch.
- `calc_nwarps` / `calc_rows_per_block` under `MMVQ_PARAMETERS_RDNA4` (`:465`, `:563`) - block shape.
  Both take `small_k` and `halve_iters`, which are where a narrow-k tensor is handled.

**`ffn_down_exps` is the narrow-k case, and the split makes it narrower.** `src/llama-model.cpp:573-583`:
`ffn_up_exps` / `ffn_gate_exps` / fused `ffn_gate_up_exps` split on `SPLIT_AXIS_1` (the output/intermediate
rows, so k stays whole), but `ffn_down_exps` splits on `SPLIT_AXIS_0` - which for down is **k**. Under
`-sm tensor` on 4 cards that leaves `moe_intermediate_size / 4 = 160` elements of k per device per row
(a 32-block type spans it in 5 blocks) and, because the reduction is over a split k, the partials must be
summed across devices - which is a plausible chunk of the 96 collectives per step per device E053 counted.
So down-projection decode matmuls are simultaneously the worst-shaped for mmvq and the only expert matmul
that costs a reduction.

**Answered by E054: mmq is not the fix.** Forcing MoE onto mmq at batch 1 (cap to 0, which also disables
the glu fusion because it consults the same table) costs **3.6% of tg** on the dev box, ~1.2% of that being
the fusion and ~2.4% mmq being the slower kernel at `n_rows == 1`. So mmvq's tuned 8-warp branch really is
the right choice on RDNA4, the `should_use_mmq` `n_experts > 0` clause is properly read as a prefill
statement, and the ~115 GB/s has to be explained inside mmvq (`calc_rows_per_block`, the `small_k` path) or
by the k-split. Caveat that survives: E054 ran `-sm none` on purpose, so the 4-card k-split case is
untested.

Two measurement traps found on the way, both worth remembering: an env boolean written as `getenv(name) !=
nullptr` is **true for `NAME=`**, which silently turned a control arm into a treatment arm; and llama-bench
CSV `avg_ns` is per repetition, not per token (128x at `-n 128`).

## What E055 found instead: mmvq's small_k is blanket-excluded for RDNA

E054 left the ~115 GB/s inside mmvq. It has a concrete cause. RDNA4 uses **8 warps** at `ncols_dst == 1`
for the types this model has (`calc_nwarps`, `mmvq.cu:465-489`), but `calc_rows_per_block`
(`mmvq.cu:563-567`) gives `MMVQ_PARAMETERS_RDNA4` the default 1 - only GENERIC/GCN/TURING/GB10 get
`small_k ? nwarps : 1` - and `should_use_small_k` discards its own computed condition for
`GGML_CUDA_CC_IS_RDNA(cc)` on the same `else if` line as the per-type NVIDIA lists, with no comment.

The condition is plainly true here: `ffn_down` at Q5_0, `k = 640` -> `blocks_per_row_x = 20`, versus
`nwarps * blocks_per_iter_1warp = 8 * 8 = 64`. So a 256-thread block cooperatively reduces a **480-byte
row**, paying a cross-warp shared-memory reduction per output value for 2.5 warps' worth of K trips. That is
exactly the case small_k was written for, disabled for the architecture with the widest blocks.

**Measured, and it transfers.** Dev box (E055): **+4.1% tg** on the 4l/512-expert dummy, **+7.4%** on the
48l-12qsa one, pp unmoved (batch 4096 is mmq territory, above `MMVQ_MAX_BATCH_SIZE`), golden bit-identical,
`MUL_MAT` / `MUL_MAT_ID` ops green. Bench box, 4x R9700, real Q4_K_M model: **+5.5% to +6.6% tg at every
depth from 4k to 131k**, well outside that box's ~1% run-to-run spread. Behind
`GGML_CUDA_MMVQ_RDNA4_SMALL_K=1`, default off, RDNA4 only.

`[>]` **Now default on** (`855a65544`); `=0` / `=off` restores the upstream shape. Which means every tg
number on this branch recorded before that commit sits ~6% low relative to the same command today -
re-baseline before comparing.

The gain is flat in depth, unlike the pool's, so the two are additive: the pool removes work that grows with
context, small_k removes per-matmul reduction work that does not.

### 4-card decode collectives: use the in-tree one-shot, not RCCL

```sh
GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_ALGO=oneshot \
  build/bin/llama-bench ... -sm tensor ...
```

+7.2% / +8.1% tg128 (d4096 / d131072) over the NCCL default on the real model, pp unchanged, host cost per
collective 15.2 -> 5.3 us. Needs all three variables: `GGML_CUDA_P2P` because peer access is gated on that
name being *present* (`ggml-cuda.cu:606`, so `=0` also enables it), `GGML_CUDA_ALLREDUCE=internal` because
Linux defaults to NCCL, and the algo because `auto` does not pick one-shot yet. Details, measurements and
the three defects it took: [experiments/plans/decode-comms-plan.md](experiments/plans/decode-comms-plan.md).

There is no RCCL tuning left to do, in case that tempts anyone: `Tree` is silently ignored
(`AllReduce | Tree = 0.0/0.0` in its own tuning table), `FC` is accepted and slower, LL plus one channel are
already chosen at 10240 B, and `VMM: no` on all four cards rules out cuMem/symmetric windows.
`RCCL_USE_AMD_SMI_LIB=1 NCCL_CUMEM_ENABLE=1` makes the collective cheaper and the run 9x slower.

Two traps from building it, both general: a peer store is visible to a **kernel** on the destination but a
`hipMemcpy` of the same address reads stale (copy engine sees DRAM), and an unfenced remote store is not
visible to either - atomicity is not visibility. Debug anything like it with
`GGML_CUDA_AR_ONESHOT_DEBUG=1` (dumps the kernel's pointers with `hipPointerGetAttributes`) and
`GGML_CUDA_AR_ONESHOT_PROBE=<n>` (round-trips the real path at init and falls back on mismatch), because
`llama-bench` mutes `GGML_LOG_INFO` without `-v` and that cost me two rounds.

Origins: `ec16a072f` ("Optimize MOE GEMV kernel for BS > 1.", #20905) added small_k and the RDNA exclusion
in the same commit - the exclusion came with the feature rather than after measuring RDNA, which is
consistent with untested rather than lost. Unproven either way, and RDNA3 is unmeasured, which is the other
half of why the clause must not simply be deleted.

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

## Split mode on 4x consumer RDNA4 (operating experience, not code)

User-reported from their own tuning, and the project's prior until measured otherwise. Treat
as experienced expectation, not verified behaviour.

- **decode: tensor split always wins.** Compute per layer runs in parallel across the cards,
  so the per-token serial path is short, at the cost of more inter-card traffic per step.
- **pp: layer split usually wins.** Layers pipeline across cards, so the transfers hide behind
  inter-layer compute overlap; the user also believes layer's comms volume is lower, but has
  not tested that. Tensor's per-layer reduce is harder to hide when each ubatch is big.
- **Measured at E006 (qwen4exp, this fork, 4x R9700): tensor won both**, including pp at
  12-32%. That contradicts the pp expectation, so either this arch breaks the pipeline-overlap
  assumption or something is stalling the graph - currently attributed to the CPU-placed
  n-gram table splitting the graph at layer 1 (`-ot per_layer_token_embd=CPU`). E007b re-tests
  pp after removing it.
- Physical reason traffic is expensive here: **Navi 48 has no Infinity Fabric**, and `lspci`
  puts all four cards under a single Zen3 root complex through two levels of PCIe switches
  (BDF 0b/10/13/19). Every inter-card byte traverses the host bridge, and there is no peer
  link to make it cheap.

## Traps

- **`docs/ops.md` and `docs/ops/*.csv` are stale and wrong for these ops** `[s]`: they mark
  `DSV4_HC_*` as unsupported on CUDA and Metal while the kernels exist, and `CUDA.csv` has no
  rows at all for `DSV4_HC*`/`GATED_DELTA_NET`/`LIGHTNING_INDEXER`/`TOPK`. Generated snapshot
  (`scripts/create_ops_docs.py`); never use it as support truth.
- `test-backend-ops` **works on gfx1201 / ROCm 6.4.4** `[v]` (dev hw **v1**; not re-verified on
  7.1.1, and ROCm 7 codegen can move op results): 1500/1500 non-FA cases and
  3973/3979 FA cases passed, no hang. The reported AMD hang is therefore not universal;
  E001's update has the stages, and the bench-box hang stays open as a box-specific symptom.
  Two real sharp edges remain: there is **no insufficient-memory skip logic** `[s]` (an
  oversized case OOM-aborts), and `-b` is an exact `strcmp` that **exits 0 having tested
  nothing** on a typo `[v]`.
- known numeric defect, not on our path: `FLASH_ATTN_EXT` fails 6 cases at
  **hsk=192/hsv=128** (gqa 8/16, permuted K/V views), err up to 0.0298 vs a 0.0005 tol `[v]`.
  qwen4exp is 256/256, which passes 142/142. Know this before re-running the FA suite and
  seeing red.
- **`test-fusion` cannot run on this backend.** It requires `ggml_backend_fusion_*`, which
  only `ggml/src/ggml-metal/ggml-metal.cpp` exports `[v]`; `--device ROCm0` exits 1 with
  "device does not export the generic fusion debugging API". That is why `tests/fusion/` has
  one CSV. So fusion coverage on ROCm has to be read from the dispatch code (see N1) or timed,
  never counted.
- **Dummy GGUFs from `test-llama-archs` abort on anything that tokenizes** `[v]`:
  `src/llama-vocab.cpp:3393` `GGML_ASSERT(tokenizer ...)` via `tokenize_input_prompts`. Use
  token-id consumers instead (`llama-bench`, `test-save-load-state`). Their weights are all
  F32 and tiny (qwen4exp: 4.80 M params, 19.24 MB), which also makes them useless as a perf
  baseline - see E002.
- qwen4exp uses `llama_memory_hybrid_idx` (`llama-model.cpp:2565`, `:2584-2589`), **not** the
  `dsa`/`msa`/`iswa` cache classes; its `set_input_qsa` is host-side index/bias construction
  with `GGML_ASSERT(r <= 64)` (`llama-memory-hybrid-idx.cpp:273-340`) `[s]` - real
  `compress_ratio = 4`, fine.
- Sparse-ness in llama.cpp is expressed **as the mask**, not as a distinct kernel
  (`src/models/qwen4exp.cpp:735-758`); the compaction is what turns "masked" into "cheaper".

## Not yet verified

- **It's `-d`, not a loop.** llama-bench already accepts comma-separated values for `-d` (and `-p`, `-n`): `-d 4096,16384,40960,131072` gives one table per row. Do not for-loop-per-depth: it is 3x as many processes and one-third the clarity.
- GPU runs do not overlap on this box (user-verified: multiple GPU runs cannot execute at the same time), so a background reviewer running gates does **not** contaminate benches - they queue, and per-run numbers stay clean.

- Whether `__GFX12__` is actually emitted for `gfx1201` (nobody ran the compiler). If it is
  not, every device-side RDNA4 path above is dead code in this build while the host-side
  heuristics still claim RDNA4. **One `amd-smi`-free check settles it**: build one TU with
  `-D__GFX12__` probing, or grep the object for the gfx12 ISA.
- Whether `-cmoe`/`-ncmoe` and `-ot` placement interacts with `ARGSORT`/`MUL_MAT_ID` support
  gates when expert weights live on CPU.
