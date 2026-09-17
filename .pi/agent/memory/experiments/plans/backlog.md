# Backlog - candidate threads, qwen4exp on RDNA4

Not experiments yet. A thread becomes an `E<nnn>` id only when it has a falsifiable
hypothesis and a deciding metric (see `PROTOCOL.md` 3.1). Statuses: `prereq` (blocks all
measurement), `probe` (needs a cheap measurement before it can become a hypothesis),
`hypothesis-ready`.

Anchors are from commit `ebbb18522`, verified by reading them. Line numbers drift -
treat function and macro names as the stable handle.

## Prerequisites

| id | thread | why it blocks |
|---|---|---|
| B0 | make the build un-shadowable: static libs or rpath, instead of relying on an `LD_LIBRARY_PATH` export | `BUILD_SHARED_LIBS=ON` + `~/.local/lib64` holding a Sep 15 llama.cpp means an unpinned run silently measures old code. Found in E001; an `export` in every command block is a memory aid, not a guarantee |
| B1 | triage `test-backend-ops` crashing on AMD GPUs | it is the tree's op-level correctness + per-op perf gate. Without it there is no cheap way to prove a kernel is both correct and faster, and rule 3 has no teeth |
| B2 | bench box: remaining unknowns (gfx target, ROCm version, PCIe topology, system RAM) | 4x Radeon AI PRO R9700 @ 32 GB is now known (`hw/bench-4x-r9700-32g.md`); the rest still gates any T2 interpretation, especially gfx, since the dev build is gfx1201-only |
| B3 | decide the ROCm version policy for the dev box | dev box is 6.4.4 (user wants 7.x); if the bench box differs, no dev-box number can be an A/B partner for a bench-box number, which collapses the two-tier design |

## Probe first (cheap, informs what to actually optimise)

| id | thread | cheap probe |
|---|---|---|
| P1 | which ops in the qwen4exp graph do not run on the GPU | `test-backend-ops support -b ROCm0` once B1 lands; failing that, a graph dump + `ggml_backend_dev_supports_op` reasoning per op. Any CPU offload dwarfs every other tuning knob |
| P2 | how many graph splits does qwen4exp cause on HIP, and where | it reduced splits in `6fe749801`, but that was measured where? splits are pure overhead at small batch and are visible in a T1 run without any weight download |
| P3 | what does `test-fusion` record for the qwen4exp row on ROCm0 vs the MTL baseline in `tests/fusion/MTL.csv` | one CSV row per (arch, moe, mode). A fusion that fires on Metal and not on ROCm is a concrete, already-quantified gap |
| P4 | is per-kernel timing available at all, and which FA family does HIP use | `GGML_HIP_EXPORT_METRICS` defaults OFF in this cache; if enabling it yields kernel timings, per-op attribution on HIP becomes possible and a lot of guesswork dies. Also determine whether the HIP build selects the `mma_f16` flash-attn family on gfx1201 or falls back to `vec` kernels - the sparse path exists only in `mma_f16`, so this gates H4b |
| P5 | ~~real qwen4exp size vs the 4-box capacity~~ **resolved** | `config.json` + Hub API fetched, and the user's deployment is Q4_K_M on GPU / Q5 PLE in RAM. Recorded in `plans/model-shape.md`. Capacity is not the constraint; bottleneck identity is |

| L1 | **is the Q5 n-gram table actually resident?** | `-lzm auto` lazy-reads any tensor over 4 GiB (`src/llama-model-loader.cpp:1088-1100`), so a ~33 GB table is probably served from the page cache by demand, not held in RAM. Check the load log for `lazy read enabled` (no rerun needed), then A/B `-lzm off` vs `auto` vs `--load-mode mmap+mlock` on `tg` stability. Cheapest high-value T2 experiment available: it is a flag flip with no code change, and irregular `tg` on a 20 M-row random gather is exactly the signature it would explain |
| L2 | PLE placement costs a graph split per ubatch | the table is the source of `ggml_get_rows(ctx0, model.per_layer_tok_embd, rows)` (`src/models/qwen4exp.cpp:1202`). On CPU that forces a CPU gather + H2D copy + graph split, and the n-gram hash is already computed host-side ("ggml has no int64 and no xor"), so PLE is a host-co-resident subsystem by construction. Question: what does that one layer cost in `tg`, and would 12 GB of the 48 GB of free VRAM be better spent keeping the table (or its hot fraction) on-GPU |
| L3 | 512 experts x 10-of-512 routing on HIP | `num_experts 512`, `num_experts_per_tok 10`, `moe_intermediate_size 640` (narrow, numerous). 2% of expert weights touched per token per layer means `tg` is a scattered-read problem. Whether HIP has anything like grouped/batched expert kernels, and what `-sm layer` imbalance does to it, is likely the largest single `tg` lever on this box |

## Hypothesis-ready (write the record, then run)

| id | thread | anchor / rationale |
|---|---|---|
| H1 | `-sm tensor` is unavailable for qwen4exp, so 4-GPU runs are stuck with layer split | `llm_arch_supports_sm_tensor` in `src/llama-arch.cpp:1130` returns **false** for `LLM_ARCH_QWEN4EXP` at `:1161`, with the upstream comment `// TODO: fix test-llama-archs`. Layer split on a hybrid model (GDN layers + full-attention layers + MoE + one PLE layer) is load-imbalanced by construction; the upstream TODO says the blocker is the dummy-model harness, not the backend. Both the gap and its cheap partial fix are in reach |
| H2 | the hyper-connection chain is under-fused on HIP | HC replaces every per-layer norm (`HC_*` tensors in `gguf-py/gguf/constants.py:672-682`, `:820-833`), so this is per-layer, every-token work. Ops landed very recently: `41abbfd59` (rms_norm+mul fusion), `37b53fd45` (hc ops, `ggml_dsv4_hc_*` at `ggml/include/ggml.h:2690-2722`). New kernels are usually correct before they are tuned |
| H3 | HC kernels may be shape-restricted, forcing fallback | the Vulkan implementation refuses `GGML_OP_DSV4_HC_PRE` unless `src[0]->ne[1] == 4` and likewise `_POST` on `src[1]->ne[1]` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:15319-15322`). If HIP has a similar restriction, real `hc_count` values that differ from the assumed one drop off the fused path. Check HIP's `supports_op` for the same shape assumptions |
| H4 | **qwen4exp pays for QSA sparsity but cannot collect it on ROCm** | The indexer and the per-layer mask rebuild are built and applied (`src/models/qwen4exp.cpp:542-691`, `:735-758`), but `build_attn_mha` is called with `n_kv_max = 0` (`:767`) instead of `top_k->ne[0]` (`:766`, commented out with `TODO: enable sparse attention when we are ready`). Without compaction the CUDA kernel iterates `n_kv = K->ne[1]` (`ggml/src/ggml-cuda/fattn-common.cuh:1131`) - full quadratic - with only a trailing-tile skip. And it is not switch-on-able: `ggml_cuda_flash_attn_ext_compact_mask` calls `GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA")` under `GGML_USE_HIP` (`ggml/src/ggml-cuda/fattn.cu:93-97`), and `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` returns `false` (`:107`). Split into H4a (cheap) and H4b (the real win) |
| H5 | PLE n-gram hashing on GPU | real numbers: `ngram_vocab_size_base 20,000,000`, `ple_embed_dim 2560`, `heads_per_ngram 8`, `ngram_size 3` -> `ple_n_heads = (3-1)*8 = 16` rows gathered per token, table ~51 B params = **28% of the model**, serving exactly **one** layer (`ple_layer_ids [2]`). The hash is computed **host-side** because ggml has no int64 and no xor (comment at `src/models/qwen4exp.cpp:1041`). See L1/L2, which now dominate this row |
| H6 | fp32-vs-fp16 output preference on RDNA4 | `ggml/src/ggml-cuda/ggml-cuda.cu:1512` picks `prefer_f32_output` for RDNA4, and `mmq.cu:248-250` / `mmvq.cu:100-126` have explicit RDNA4 tile/param paths. These are global decisions with per-model consequences - a per-model override is a plausible, small, measurable win |
| H7 | VMM disabled | `GGML_HIP_NO_VMM` defaults **ON** (i.e. VMM off) in this cache, and the device banner prints `VMM: no`. That changes allocation and fragmentation behaviour, which is exactly what a 16 GB single card and a 4-card layer split both stress |
| H8 | `GGML_HIP_RCCL=OFF` in this build | if multi-GPU ends up needing collectives rather than pure layer split, RCCL is not compiled in. Cheap to flip, but only worth a record once B2 says what the topology is |

| H4a | stop paying the indexer + mask-rebuild tax while compaction is unavailable | The mask rebuild is `ggml_fill(-INF)` + `ggml_set_rows` + `ggml_add` over `[n_kv, n_batch, 1, n_stream]` per full-attention layer per ubatch - memory traffic scaling with context, for a mask whose interior structure the kernel then ignores. `src/models/qwen4exp.cpp:566` already shrinks what is uploaded to `1/ratio` of the cells, so what remains is on-device. Measure `pp` against today on a synthetic model at growing context, semantics held identical. Obsolete the moment H4b lands: a stepping stone, not a destination |
| H4b | port `flash_attn_mask_to_sparse_indices` to HIP, then flip qwen4exp to pass `top_k->ne[0]` | The NVIDIA kernel is self-contained warp-ballot compaction (`ggml/src/ggml-cuda/fattn.cu:10-32`), `values_per_lane = 8`, uses `uint32_t(1) << lane` so it assumes `WARP_SIZE == 32` - true on this box (`Wave Size: 32`). Its `ggml_cuda_pdl_sync()` / `ggml_cuda_pdl_lc()` programmatic-dependent-launch hooks are NVIDIA-only and need a non-PDL equivalent. Vulkan's `flash_attn_sparse_compact.comp` is a second reference. Gated on P4 |

## Explicitly out of scope for now

- Vulkan, SYCL, OpenCL, CPU-only paths. Vulkan appears above only as a reference for what
  a fused kernel looks like.
- Quality/accuracy tuning (quant selection studies, chat templates, samplers). This
  branch is about throughput and memory on RDNA4.
- Upstreaming anything. Local branch; if something later becomes a PR, it gets its own
  discussion with the maintainer first, per `AGENTS.md`.

## Dead ends

(empty by construction - promote a row here from a `dead-end` verdict in `INDEX.md`)
