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
| B4 | **pull in the bench box's fork changes** (user-requested) | The fork at `c9a59ef73` carries three things this branch lacks, each already proven useful on this exact hardware: **(1) RDNA4 MMQ fixes**, **(2) a custom AllReduce** (RCCL does not work on that setup - and here `GGML_HIP_RCCL=OFF` plus the `#ifndef GGML_USE_HIP` guard on the "rebuild with NCCL" warning means the fallback to internal AllReduce is *silent*), **(3) `-sm tensor` enabled for qwen4exp**. Ordering matters: the user reports the P2P parts **conflict with recent upstream changes**, so MMQ should land first and the AllReduce last. Measure after each - three separate A/Bs, not one unresolvable blob |

## Likely bottlenecks (this is where the effort should go)

| id | thread | why |
|---|---|---|
| L1 | **is the Q5 n-gram table actually resident?** | `-lzm auto` lazy-reads any tensor over 4 GiB (`src/llama-model-loader.cpp:1088-1100`) `[v]`, so a ~33 GB table is probably served from the page cache on demand, not held in RAM. Check the existing load log for `lazy read enabled`, then A/B `-lzm off` / `auto` / `--load-mode mmap+mlock` on `tg` stability. Zero code change, no rerun to diagnose, and irregular `tg` over a 20 M-row random gather is exactly its signature |
| L2 | PLE placement is *suspected* of costing tg - unresolved | the table is the `ggml_get_rows` source at `src/models/qwen4exp.cpp:1202`. On CPU: host gather + H2D copy + split, and the n-gram hash is already host-side (no int64/xor in ggml). **Whether that is what costs 35 ms/token is unknown** - the user's own read is "hard to say". Putting the table on GPU is not an option (mirrored, ~30 GB/card), so the earlier "keep it on GPU" framing here was wrong. Next step is measuring faults, not building: `ple-prefetch.md` method 1 = `iostat` during a decode run, zero code |
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
| H1 | ~~`-sm tensor` unavailable for qwen4exp forces layer split~~ **revised by E005** | This branch *throws* for `-sm tensor` on qwen4exp: `llm_arch_supports_sm_tensor` returns false (`src/llama-arch.cpp:1161`) and `llama_model_create` raises `LLAMA_SPLIT_MODE_TENSOR not implemented` (`src/llama-model.cpp:358`). **But the user runs tensor split on the bench box**, so their fork enables it - and merging forward will break their command line until that patch comes along (B4). The upstream guard is test-driven (`// TODO: fix test-llama-archs`), i.e. the blocker is the dummy model, not the backend. This row previously claimed layer split was forced; it was not, and that error cost a detour. **E006: tensor wins tg, as the user expected.** For pp the general rule is the *opposite* - layer usually wins, because layers pipeline across cards and the transfers hide behind inter-layer compute overlap - so this run's layer-loses-pp result is an anomaly, not a rule, and it is now read as a symptom of the CPU split (F3, E007b). The guard stays an obstacle to clear in B4 |
| H2 | the hyper-connection chain is under-optimised on HIP | HC replaces every per-layer norm, so it is per-layer and every-token in both modes. Only `_PRE`(gated) and `_POST`(comb=null) are emitted `[x]`; both are f32-only `[v]`; and `rms_norm+mul` fusion was only just enabled (`41abbfd59`). New kernels are usually correct before they are tuned. Bounded by N5 |
| H4a | stop paying the indexer + mask-rebuild tax while compaction is unavailable | mask rebuild is `fill(-INF)` + `set_rows` + `add` per full-attn layer per ubatch (`src/models/qwen4exp.cpp:735-758`) `[v]` - traffic scaling with context for a mask whose interior the kernel ignores. `:566` already trims the upload to `1/ratio` of cells. Obsolete the moment H4b lands |
| H4b | port the mask compaction to HIP, then flip `n_kv_max` | narrowed by the survey to: one warp-ballot kernel (`fattn.cu:10-89`, `WARP_SIZE == 32` which gfx1201 has, but `ggml_cuda_pdl_*` are NVIDIA-only), the `#if !defined(GGML_USE_HIP)` compile guards (`:92-96`, `:109-113`, `:133-140`) `[v]`, and the call site (`qwen4exp.cpp:767`) `[v]`. **Updated by E001: flipping the call site first is inert, not a safe first step** - sparse cases already report SUPPORTED and compute dense, so results and cost are unchanged either way. Effect size: `indexer_budget 2048` of 262,144 context, on 12 of 48 layers, pp-only per P4 |
| H5 | PLE n-gram hashing | `ple_n_heads = (3-1)*8 = 16` gathers per token from a ~20 M x 2560 table = ~51 B params = 28% of the model, serving one layer, hashed host-side. Superseded in priority by L1/L2 |
| H6 | fp32 output preference on RDNA4 | `prefer_f32_output` is forced on for RDNA4 (`ggml-cuda.cu:1512`, `:1514`) `[v]`, and `mmvq.cu:417-492` has an RDNA4-only `nwarps` whitelist `[s]`. Confirmed as real, still unmeasured: a per-model override is a plausible small win |
| H7 | VMM disabled | `GGML_HIP_NO_VMM` defaults ON, and the `VMM: no` banner is just that flag, not a device query `[s]`. Allocation/fragmentation behaviour differs from a CUDA default - relevant to both the 16 GB box and a 4-card split |
| H8 | `GGML_HIP_RCCL=OFF` | and the "rebuild with NCCL" warning is `#ifndef GGML_USE_HIP`, so without RCCL the fallback to internal AllReduce is **silent** `[s]`. Only worth a record once B2 reports the topology |

## Next runs (ids reserved, flag-only, bench box)

| id | change | deciding observation | source |
|---|---|---|---|
| ~~E007~~ | ~~drop `-ot per_layer_token_embd=CPU`~~ **killed by the user, confirmed in code** | under `-sm tensor` the PLE table is **mirrored, not split** (`src/llama-model.cpp:513-515`), so ~30 GB becomes ~30 GB *per card* = ~120 GB of a 128 GB box. Not a tuning question. Replaced by `ple-prefetch.md` |
| ~~E008~~ | ~~today's config + `GGML_CUDA_DISABLE_GRAPHS=1`~~ **done, answer: no** | capture is active: disabling graphs costs 7% of tg and ~7% of deep pp, so the CPU split does not defeat it. Also bounds total launch-submission cost at ~2.7 ms/token. See E008 |
| E008b | `-v` load log + `free -h` / `vmstat 1` / `iostat -x 1` during steady decode | still worth it: system RAM is unknown, and the PLE fault ceiling only holds if the box is not swapping | E005 open q2 |
| E007b | ~~repeat the `-sm` pp A/B after dropping `-ot`~~ **deprioritised** | E007 is impossible (mirrored table) and E008 killed the capture chain, so there is no placement change left to re-test pp against | E006 |
| E009 | `GGML_SCHED_DEBUG_REALLOC=1` on the same command line | runtime `getenv` (`ggml/src/ggml-backend.cpp:1865`), no rebuild; its comment names this exact pathology - graph reallocated even though its size did not change. If decode reallocates every step, per-step host allocation is where the missing ~33 ms lives |
| E010 | `LLAMA_GRAPH_REUSE_DISABLE=1` (`src/llama-context.cpp:279`) | inverse probe: if killing graph *reuse* costs far more than killing graphs cost (7%), the time is in per-step rebuild rather than replay |
| E011 | `llama-batched-bench ... -npp 512 -ntg 128 -npl 1,2,4` (**not** `llama-bench`, which has no `-np`; in batched-bench the sweep flag is `-npl`, and `-np` is a separate common arg for sequences to decode) | cleanest discriminator, needs no debug hooks: if aggregate tg scales **super**-linearly with `npl`, per-step host work is being amortised over more tokens and the host path is confirmed. Sub-linear means it is GPU work and the host-path reasoning dies |
| E012 | `-sm row` as a third point, everything else as E005 | `-sm` has four modes (`none,layer,row,tensor`) and E006 compared only the two endpoints. `row` splits weights across GPUs (parallelized) but keeps KV on the main GPU, so it **decomposes** tensor split into weights-split + KV-split: if row ~ tensor for tg, KV splitting is implicated; if row ~ layer, weight splitting is what was helping. Untested by the user |
| E013 | sweep `-d` (real depth) at fixed `-p`/`-n`, e.g. `-d 512,4096,16384,40960,131072` | **justification corrected**: `-d` does fill the KV (`llama-bench.cpp:2408-2433`), so E005/E008 are single-depth measurements at ~40 k, not shallow ones. What is missing is the *curve*: depth response separates attention/KV cost from per-step fixed cost, and only the curve can say how much of the 35.5 ms is attention. Cheapest way to make H4b's value quantitative at 262 k | E005 |

## Code-level items

| id | thread | anchor |
| F3 | **retracted as written: the CPU split does NOT defeat graph capture** | E008 showed capture is active (7% of tg, ~7% of deep pp). What survives: `-ot per_layer_token_embd=CPU` still forces a host gather + H2D + graph split at layer index 1 every step, and a split graph means several subgraphs each needing host-side build/alloc - now the leading suspect precisely because graph replay cannot remove it (E009-E011). Launch cost itself is bounded at ~2.7 ms/token by E008 |
| F1 | **the lazy table is never prefetched, in any load mode**, so there is no way to warm it | `src/llama-mmap.cpp`: the `POSIX_MADV_WILLNEED` loop iterates `ranges_complement(lazy_ranges, ...)` (`:500-502`) - i.e. it prefetches everything *except* the lazy ranges; `MAP_POPULATE` is skipped with the comment "MAP_POPULATE would fault in the lazy ranges too" (`:481`); and the lazy ranges get `POSIX_MADV_RANDOM` (`:508-510`), which turns kernel read-ahead **off** for them. Also `if (numa) { prefetch = 0; }` (`:473`) plus a whole-file `MADV_RANDOM` (`:516-522`). Asked and answered: `-lm none` is not the cause, since no mode warms the table. What is missing is an opt-in pre-warm of the lazy ranges (a background `WILLNEED` during tensor upload), or a `-lzm` variant for "resident but lazy-mapped". With `ple_n_heads = 16` and `MADV_RANDOM`, that is up to ~16 faults per token on a ~30 GB range. **Design options, including the user's two prefetch ideas and why they split by mode: see `ple-prefetch.md`** |
| F2 | `size_label` is cosmetic and derived from the **repo/file name** (`gguf-py/gguf/metadata.py:314-328`), never computed from parameters | it reported `A3B` for a ~6 B-active model and misled this analysis for a full round trip. Derive active params yourself; `llama-bench`'s `model_type` column inherits the label. Related consistency check that *is* trustworthy: the GGUF reports 176.94 B params vs 179.99 B in HF bf16, a ~3.05 B gap consistent with the exporter dropping the 1-layer MTP block (`conversion/qwen4exp.py:21-22`) - one MoE layer plus its attention/embedding weight |
| F3 | **`-ot per_layer_token_embd=CPU` is now the prime suspect for tg** | it puts a ~30 GB table on a CPU buffer, forcing a host gather + H2D copy + **graph split at layer index 1 every step**, with the n-gram hash host-side regardless (no int64/xor in ggml). E006 showed tg is insensitive to split mode yet ~12-44x above its floor - consistent with a shared serial stall, and a CPU split mid-graph is the obvious candidate. Sharper form: **the split may be what prevents HIP graph capture for the whole graph**, in which case decode degrades to eager per-node submission from one host thread (a few thousand nodes/token -> tens of ms) and the 9% util is the GPUs waiting to be told what to do. Test via E007/E008 before writing code |
| T1 | implement `ggml_backend_fusion_*` for the CUDA/HIP backend | would unlock `test-fusion` on `ROCm0`, i.e. per-arch fusion counts and a `CUDA.csv` baseline (upstream ships only MTL). Only worth it if N1 (IMROPE fusion gap) or H2 need *counting* rather than timing. It is tooling, not a perf win - P3 died on this. **Demoted by E005**: counting fusions is not the bottleneck question while the box sits at 9% utilisation |

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
