# Backlog - candidate threads, qwen4exp on RDNA4

Not experiments yet. A thread becomes an `E<nnn>` id only when it has a falsifiable
hypothesis and a deciding metric (see `PROTOCOL.md` 3.1).

Anchors at commit `ebbb18522`. **[v]** = read directly by me, **[s]** = from the `scout-1`
survey not re-read, **[x]** = corrected after verification. See `../rdna4-rocm-build`
memory for the full HIP/RDNA4 backend picture, and `model-shape.md` for real dims.

## Prerequisites

| id | thread | why it blocks |
|---|---|---|
| B0 | make the build un-shadowable: **rpath**, since `GGML_STATIC` is a hard `FATAL_ERROR` on the HIP path | `BUILD_SHARED_LIBS=ON` + `~/.local/lib64` holding a Sep 15 llama.cpp means an unpinned run silently measures old code. Found in E001. `CMAKE_BUILD_RPATH=$PWD/build/bin` is the one-line fix |
| B1 | ~~triage `test-backend-ops` crashing on AMD GPUs~~ **closed on the dev box** | it does not crash here: 1500/1500 non-FA and 3973/3979 FA cases pass on gfx1201 / ROCm 6.4.4, so rule 3 has a working gate (E001 update). The bench-box hang the user remembers is real but box-specific - candidates are the 4-GPU config, its ROCm version, or its code state, all B2 unknowns. Two sharp edges to keep in mind: no insufficient-memory skip logic (oversized case OOM-aborts) `[s]`, and `-b` is an exact `strcmp` that exits 0 having tested nothing on a typo `[v]` |
| B2 | bench box: ROCm version, PCIe topology, system RAM | gfx target is now known: **gfx1201 on both boxes** (user-confirmed), so a dev build is ISA-valid there. What remains decides whether numbers are comparable, not whether binaries run |
| B3 | decide the ROCm version policy for the dev box | dev box 6.4.4, user wants 7.x. Only one cmake gate exists (ROCm >= 6.1) `[v]` and there is no `ROCM_VERSION` in-code gating `[s]`, so an upgrade is lower-risk to code paths than assumed - but still a comparability break for measurements |

## Likely bottlenecks (this is where the effort should go)

| id | thread | why |
|---|---|---|
| L1 | **is the Q5 n-gram table actually resident?** | `-lzm auto` lazy-reads any tensor over 4 GiB (`src/llama-model-loader.cpp:1088-1100`) `[v]`, so a ~33 GB table is probably served from the page cache on demand, not held in RAM. Check the existing load log for `lazy read enabled`, then A/B `-lzm off` / `auto` / `--load-mode mmap+mlock` on `tg` stability. Zero code change, no rerun to diagnose, and irregular `tg` over a 20 M-row random gather is exactly its signature |
| L2 | PLE placement costs a graph split per ubatch | the table is the `ggml_get_rows` source at `src/models/qwen4exp.cpp:1202`. On CPU: host gather + H2D copy + split, and the n-gram hash is already host-side (no int64/xor in ggml). What does that one layer cost in `tg`, and would some of the ~48 GB free VRAM be better spent keeping the table (or its hot fraction) on-GPU? |
| L3 | 10-of-512 expert routing on HIP | `num_experts 512`, `per_tok 10`, `moe_intermediate_size 640`. ~2% of expert weights touched per token per layer, so `tg` is a scattered-read problem. Good news from the survey: `should_use_mmq` is **unconditionally true on RDNA4** (`mmq.cu:380-382`) `[s]` and MMQ tile selection is already expert-aware (`mmq.cu:248-251`) `[s]`, so the machinery is tuned for this - the open question is `-sm layer` imbalance across 4 cards |
| L4 | 16 GB of fp32 recurrent state | `mamba_ssm_dtype: float32` x 36 linear-attention layers with key 2048 / value 6144 wide. State size is a fixed VRAM tax that competes with the 262 k context, and it resists fp16 tricks |

## Closed by the survey (do not re-probe)

| id | was | now |
|---|---|---|
| P1 | which ops fall off the GPU | **now measured, not merely reasoned:** `test-backend-ops support -b ROCm0` over 13 families -> 9906 supported / 2384 unsupported cases, and the non-FA `test` run passed 1500/1500 `[v]`. Every op in the graph has a real HIP kernel. Remaining escape hatches: elementwise contiguity gates, and `ARGSORT` needing `ne[0] <= 1024` `[v]` - real `num_experts = 512`, so the router stays on GPU |
| P4 | which FA family does gfx1201 get | **`mma_f16` for prompt processing, not for decode.** `amd_wmma_available` + DK 256 + `gqa_ratio_eff 4` gives threshold `Q->ne[1]*4 > 16` (`ggml/src/ggml-cuda/fattn.cu:667-671` `[v]`, `(256,256,*)` instances exist `[v]`). Decode of 1 token falls to tile/vec. Since sparse FA lives only in `mma_f16`, H4b is a **pp-only** win - which is where long-context cost is anyway |
| H3 | HC kernels may be shape-restricted | **not on HIP.** The gate is dtype-only, all-F32, no shape restriction (`ggml-cuda.cu:5492-5501`) `[v]`. The `ne[1] == 4` rule I feared is Metal/Vulkan's. Replaced by N1 below |
| P5 | does the model fit | capacity is not the constraint on this box; see `model-shape.md` |

## New, from the survey

| id | thread | anchor |
|---|---|---|
| N1 | **`RMS_NORM+MUL+ROPE` fusion is rejected for qwen4exp because it is IMROPE** | the 3-node fusion accepts only `GGML_ROPE_TYPE_NORMAL`/`NEOX` (`ggml/src/ggml-cuda/ggml-cuda.cu:2734-2735`) `[v]` and `llama-model.cpp:3063-3070` gives this arch `IMROPE` `[s]`. Self-contained, clearly-scoped fusion gap on 12 full-attention layers |
| N2 | a quantized KV cache may not be usable | `SET_ROWS` dst whitelist excludes **Q4_K/Q5_K/Q6_K** `[s]`, and qwen4exp writes KV through `cpy_k`/`cpy_v` = `set_rows`. If anyone tries a Q4_K KV cache to fit 262 k context, that is a support failure, not a slowdown |
| N3 | `GGML_CUDA_DEVICES` makes multi-device testable on the 16 GB box | it exposes N *virtual* devices round-robined over physical GPUs (`ggml-cuda.cu:235-259`) `[v]`. So `-sm layer` split behaviour, per-device balance and split counts become **T1**, not T2-only. Directly upgrades H1 |
| N4 | verify `__GFX12__` is actually emitted for gfx1201 | device-side RDNA4 paths hinge on `vendors/hip.h:215-217` (`__GFX12__ -> RDNA4`) `[s]`, never compiled-and-checked. If absent, every `#if defined(RDNA4)` device branch is dead while host-side `IS_RDNA4(cc)` still claims RDNA4 - a half-tuned build that looks fine |
| N5 | HC inputs are F32-only | the gate requires f32 for all HC operands `[v]`, so any fusion or cast that lands f16 there silently leaves the fused path. Constraint on H2, not an experiment |

## Hypothesis-ready (write the record, then run)

| id | thread | anchor / rationale |
|---|---|---|
| H1 | `-sm tensor` unavailable for qwen4exp forces layer split on the 4-GPU box | `llm_arch_supports_sm_tensor` returns false for `LLM_ARCH_QWEN4EXP` (`src/llama-arch.cpp:1161`), upstream `// TODO: fix test-llama-archs`. Layer split on a hybrid stack (36 GDN + 12 QSA + MoE + 1 PLE layer) is imbalanced by construction. Now partly testable locally via N3 |
| H2 | the hyper-connection chain is under-optimised on HIP | HC replaces every per-layer norm, so it is per-layer and every-token in both modes. Only `_PRE`(gated) and `_POST`(comb=null) are emitted `[x]`; both are f32-only `[v]`; and `rms_norm+mul` fusion was only just enabled (`41abbfd59`). New kernels are usually correct before they are tuned. Bounded by N5 |
| H4a | stop paying the indexer + mask-rebuild tax while compaction is unavailable | mask rebuild is `fill(-INF)` + `set_rows` + `add` per full-attn layer per ubatch (`src/models/qwen4exp.cpp:735-758`) `[v]` - traffic scaling with context for a mask whose interior the kernel ignores. `:566` already trims the upload to `1/ratio` of cells. Obsolete the moment H4b lands |
| H4b | port the mask compaction to HIP, then flip `n_kv_max` | narrowed by the survey to: one warp-ballot kernel (`fattn.cu:10-89`, `WARP_SIZE == 32` which gfx1201 has, but `ggml_cuda_pdl_*` are NVIDIA-only), the `#if !defined(GGML_USE_HIP)` compile guards (`:92-96`, `:109-113`, `:133-140`) `[v]`, and the call site (`qwen4exp.cpp:767`) `[v]`. **Updated by E001: flipping the call site first is inert, not a safe first step** - sparse cases already report SUPPORTED and compute dense, so results and cost are unchanged either way. Effect size: `indexer_budget 2048` of 262,144 context, on 12 of 48 layers, pp-only per P4 |
| H5 | PLE n-gram hashing | `ple_n_heads = (3-1)*8 = 16` gathers per token from a ~20 M x 2560 table = ~51 B params = 28% of the model, serving one layer, hashed host-side. Superseded in priority by L1/L2 |
| H6 | fp32 output preference on RDNA4 | `prefer_f32_output` is forced on for RDNA4 (`ggml-cuda.cu:1512`, `:1514`) `[v]`, and `mmvq.cu:417-492` has an RDNA4-only `nwarps` whitelist `[s]`. Confirmed as real, still unmeasured: a per-model override is a plausible small win |
| H7 | VMM disabled | `GGML_HIP_NO_VMM` defaults ON, and the `VMM: no` banner is just that flag, not a device query `[s]`. Allocation/fragmentation behaviour differs from a CUDA default - relevant to both the 16 GB box and a 4-card split |
| H8 | `GGML_HIP_RCCL=OFF` | and the "rebuild with NCCL" warning is `#ifndef GGML_USE_HIP`, so without RCCL the fallback to internal AllReduce is **silent** `[s]`. Only worth a record once B2 reports the topology |

## Hygiene notes

- **`docs/ops.md` / `docs/ops/*.csv` are stale and actively wrong for these ops** `[s]`:
  they mark `DSV4_HC_*` unsupported on CUDA and Metal while the kernels exist, and
  `CUDA.csv` has no rows at all for `DSV4_HC*`/`GATED_DELTA_NET`/`LIGHTNING_INDEXER`/`TOPK`.
  Never cite them as support evidence.
- qwen4exp uses `llama_memory_hybrid_idx`, **not** the `dsa`/`msa`/`iswa` cache classes `[s]`.
- Sparse-ness is expressed *as the mask*; compaction is what makes a mask cheaper to
  iterate. No separate sparse kernel exists in ggml-cuda `[s]`.
- `FLASH_ATTN_EXT` has **6 known numeric failures at hsk=192/hsv=128** (gqa 8/16, permuted
  K/V views, err up to 0.0298 vs 0.0005 tol) `[v]`. Not our shape - do not chase it, but do
  not mistake it for a regression you introduced when re-running the FA suite.
- Empirical FA support at our shape: **hsk/hsv 256/256 is 142/142 supported on `ROCm0`**
  (f16/q4_0/q8_0 KV) `[v]`. Zero support: 96/64, 128/64, 192/192, 64/128 - never assume a
  mismatched DK/DV pair works.

| T1 | implement `ggml_backend_fusion_*` for the CUDA/HIP backend | would unlock `test-fusion` on `ROCm0`, i.e. per-arch fusion counts and a `CUDA.csv` baseline (upstream ships only MTL). Only worth it if N1 (IMROPE fusion gap) or H2 need *counting* rather than timing. It is tooling, not a perf win, and it is a real chunk of work - P3 died on this |

## Explicitly out of scope for now

- Vulkan, SYCL, OpenCL, CPU-only paths. Vulkan appears above only as a reference for what a
  fused kernel looks like.
- Quality/accuracy tuning (quant selection studies, chat templates, samplers). This branch is
  about throughput and memory on RDNA4.
- Upstreaming anything. Local branch; anything that later becomes a PR gets its own discussion
  with the maintainer first, per `AGENTS.md`.

## Dead ends

| id | what died | why it is still useful |
|---|---|---|
| E002 | a synthetic qwen4exp model as a pp/tg baseline | the 19 MB F32 model fits in cache, so pp/tg measure harness overhead, not bandwidth. Killed cheaply and on purpose, which is the point: nobody should treat `tg` 325 t/s as a reference number. Replacement is E004 (config-shaped dummy) or T2-only |
| P3 | `test-fusion` counts on ROCm0 | the fusion debug API is Metal-only `[v]`; the tool refuses to run rather than returning empty numbers. See T1 if the signal is ever worth building |
