# INDEX - qwen4exp / RDNA4 experiments

Append-only ledger. One row per experiment. Newest at the bottom (ids never reuse).
Detail lives in `runs/`; this table is for scanning and for answering "did we already
try that?".

| id | date | tier | machine | hypothesis (one line) | headline result | verdict | record |
|---|---|---|---|---|---|---|---|
| E001 | 2026-09-17 | T1 | dev-rx9070-16g | the T1 harness (dummy models + fusion counts + bench) runs on ROCm0 | loader trap was the whole story. `test-backend-ops` passes (1500/1500 non-FA, 3973/3979 FA); 111 dummy models generate; qwen4exp graph runs on GPU incl. `-fa 1`; **`test-fusion` is impossible here - the fusion debug API is Metal-only** | done | [runs/E001-t1-harness-viability.md](runs/E001-t1-harness-viability.md) |
| E002 | 2026-09-17 | T1 | dev-rx9070-16g | a synthetic qwen4exp model gives a usable pp/tg baseline | **no**: 19 MB F32 fits in cache, so pp/tg measure launch + input-path overhead, not the real bottleneck | dead-end | [runs/E002-synthetic-baseline.md](runs/E002-synthetic-baseline.md) |
| E003 | - | T1 | dev-rx9070-16g | qwen4exp pays a context-scaling QSA tax for sparsity it cannot collect on ROCm | - (instrument now blocked on E004; `n_kv_max` flip is inert, see E001) | planned | [runs/E003-qsa-tax.md](runs/E003-qsa-tax.md) |
| E004 | - | T1 | dev-rx9070-16g | a config-shaped synthetic model (real head_dim 256 / 24-2 heads / hc_lowrank 320 / budget 2048, fewer experts) is a usable pp instrument | - | planned | (record not yet written) |
| E005 | 2026-09-17 | **T2** | bench-4x-r9700-32g | real qwen4exp Q4_K_M on 4x R9700 gives us a baseline | pp512/4k/8k = 397/512/547 t/s, tg128 = 28.2 t/s; **~9% GPU util**; **not a baseline** - user's fork, different ROCm | done | [runs/E005-real-baseline.md](runs/E005-real-baseline.md) |
| E006 | 2026-09-17 | T2 | bench-4x-r9700-32g | `-sm tensor` costs tg via per-layer cross-GPU collectives (no Infinity Fabric on Navi 48) | **refuted by sign**: layer split does ~zero collectives and is *slower* (tg 24.7 vs 28.2); also refutes bandwidth-bound (predicted 0.25 ratio, measured 0.88). tg is serial and mode-independent | done | [runs/E006-split-mode.md](runs/E006-split-mode.md) |
| E007 | - | T2 | bench-4x-r9700-32g | move the n-gram table onto VRAM to avoid the host gather | **impossible**: the PLE path is *mirrored* under `-sm tensor` (`src/llama-model.cpp:513-515`), so ~30 GB becomes ~30 GB/card on a 128 GB box | dead-end | folded into [plans/ple-prefetch.md](plans/ple-prefetch.md) |
| E008 | 2026-09-17 | T2 | bench-4x-r9700-32g | the CPU-placed table might defeat HIP graph capture | **no** - capture is active: graphs off costs 7% of tg and ~7% of deep pp. Bounds total launch-submission cost at ~2.7 ms/token | done | [runs/E008-graph-capture.md](runs/E008-graph-capture.md) |
| E009 | 2026-09-17 | T2 | bench-4x-r9700-32g | the graph may be reallocated at unchanged size every step | **no** - the hook GGML_ABORTs when it fires and the run finished clean; same-size realloc ruled out | done | [runs/E009-graph-realloc.md](runs/E009-graph-realloc.md) |
| E010 | 2026-09-17 | T2 | bench-4x-r9700-32g | graph reuse may already be broken for this model | **no, best number yet**: reuse works, worth ~22 ms/step (tg 28.20->17.36; pp512 +19.6 ms on its single build); pp insensitive, decode collapses | done | [runs/E010-graph-reuse.md](runs/E010-graph-reuse.md) |
| E011 | 2026-09-17 | T2 | bench-4x-r9700-32g | if tg scales super-linearly with concurrency the cost is per-step host work | **yes**: ~28 ms/step is fixed regardless of tokens (B=1/2/4 -> 32.4/36.7/52.5 ms per step); ~90% of a decode token is per-step cost. Bonus: 40k depth costs only ~10% of tg, bounding H4b | done | [runs/E011-concurrency.md](runs/E011-concurrency.md) |
| E014 | - | T2 | bench-4x-r9700-32g | `-lzm off` + `-ot ...=CPU` gives a resident table with no demand paging; if tg improves, faults are part of the fixed ~28 ms | - | planned | [runs/E011-concurrency.md](runs/E011-concurrency.md#found-in-the-header-of-this-log--ot-has-never-done-anything) |
| E015 | 2026-09-17 | T2 | bench-4x-r9700-32g | scheduler op-offload may be the fixed per-step cost | **no** - tg 28.14 vs 28.20, pp 547.5 vs 546.8, CSV confirms `no_op_offload=1` applied | dead-end | [runs/E015-no-op-offload.md](runs/E015-no-op-offload.md) |

## Comparability breaks

Any row here means numbers on either side of it are not valid A/B partners.

| date | machine | change | invalidated |
|---|---|---|---|
| 2026-09-17 | both | **the two boxes are not the same stack**: bench is ROCm 7.15.0 / Fedora 44 / kernel 7.2.5, dev is ROCm 6.4.4 / Fedora 43 | no dev-box number may be an A/B partner for a bench-box number, and vice versa. E005's numbers additionally come from a **fork** (`c9a59ef73`) with RDNA4 MMQ fixes, a custom AllReduce, and qwen4exp tensor-split enablement that this branch does not have |

## Machine profiles

| profile | role | state |
|---|---|---|
| [hw/dev-rx9070-16g.md](hw/dev-rx9070-16g.md) | T1: build, op-level, synthetic models | v1, characterised. ROCm 6.4.4 (upgrade wanted) |
| [hw/bench-4x-r9700-32g.md](hw/bench-4x-r9700-32g.md) | T2: real weights, end-to-end, multi-GPU | v1: gfx1201 confirmed, 4x32 GB = 128 GB. Runs Q4_K_M on GPU + Q5 PLE table in RAM. ROCm ver / topology / sys RAM still unknown |

## Open threads (not yet experiments)

Tracked in [plans/backlog.md](plans/backlog.md). Promote a thread to an `E<nnn>` id
only when it has a falsifiable hypothesis written down.

Substrate reference: [plans/model-shape.md](plans/model-shape.md) - real `qwen4exp`
dimensions vs what the synthetic dummy uses, and the list of places where the dummy's
value sits on a kernel-selection boundary. Needs the HF `config.json` (text only, no
weights) to fill. **Filled 2026-09-17.**

Design reference: [plans/ple-prefetch.md](plans/ple-prefetch.md) - the n-gram table
prefetch/cache options (warm page cache; VRAM row cache with an S3-FIFO/SIEVE-class policy;
why the whole table cannot go on GPU), and what has to be measured before any of it is built.

## Findings that are not experiments

Things established while setting up, recorded here because they change what is worth
measuring at all. Detail in the topic memories.

- **Sparse FA does not exist on ROCm, and qwen4exp still pays for it.** The
  indexer plus a per-layer mask rebuild run on every full-attention layer, but
  `build_attn_mha` passes `n_kv_max = 0`, so the kernel iterates the whole KV extent.
  Switching it on is not possible on this backend today:
  `ggml_cuda_flash_attn_ext_compact_mask` does `GGML_ABORT("sparse flash attention is only
  supported on NVIDIA CUDA")` under `GGML_USE_HIP`
  (`ggml/src/ggml-cuda/fattn.cu:93-97`). Drives E003, H4a, H4b.
- **`qwen4exp` is a 180 B MoE, and 28% of it is one hash-gathered embedding table.**
  51.2 B params of PLE n-gram table serving a single layer, run as Q5 in host RAM by the
  user; `-lzm auto` lazy-reads any tensor over 4 GiB
  (`src/llama-model-loader.cpp:1093`), so "in RAM" may mean "page-faulted on demand".
  Backlog L1/L2. Full shape in [plans/model-shape.md](plans/model-shape.md).
- **`test-fusion` does not exist for this backend.** The generic fusion debugging API
  (`ggml_backend_fusion_*`) is implemented only in `ggml/src/ggml-metal/ggml-metal.cpp` `[v]`,
  so `--device ROCm0` refuses to run and `tests/fusion/` has exactly one baseline CSV. This
  kills P3 and removes the natural instrument for N1 (the IMROPE fusion gap). Backlog T1.
- **A synthetic qwen4exp model is not a perf baseline** (E002, dead-end): 19 MB F32 fits in
  cache, so pp/tg report harness overhead. It *is* a valid structural instrument - the graph
  loads and runs on `ROCm0`, with and without `-fa 1`.
- **Dummy GGUFs abort anything that tokenizes** - `llama-vocab.cpp:3393` assert, no
  tokenizer in the file. Use token-id consumers (`llama-bench`, `test-save-load-state`).
- **`-sm tensor` throws on this branch for qwen4exp but is what the bench box actually
  runs**: `llm_arch_supports_sm_tensor` returns false (`src/llama-arch.cpp:1161`) and
  `llama_model_create` raises (`src/llama-model.cpp:358`); the user's fork enables it, and
  E006 measured tensor split winning decode. So this is an obstacle for B4, not a hardware
  limitation - and I previously had it backwards, claiming the box was stuck on layer split.
- The dev box's `~/.local/lib64` shadows the working tree for every dynamic binary, and it
  already produced two false failures. Drives B0.

- **`test-backend-ops` is not broken on AMD.** On gfx1201 / ROCm 6.4.4 it passed 1500/1500
  non-FA and 3973/3979 FA cases with no hang (E001 update). The 6 failures are all
  `hsk=192/hsv=128`, a shape qwen4exp does not use. So B1 is closed for the dev box and the
  bench-box hang is a box-specific symptom needing a different explanation - 4-GPU config,
  ROCm version, or code state, all B2 unknowns.
- **Flipping `n_kv_max` on ROCm is inert, not risky.** Sparse-shaped FA cases report
  `SUPPORTED` and then silently compute dense, because `use_sparse` is false on HIP and
  `compact_mask` is never reached. H4b is kernel work; it cannot be staged from the model
  side. See E001 stage 3.
- **Sparse FA support at qwen4exp's shape is empirically fine on ROCm0:** 142/142
  `FLASH_ATTN_EXT` cases at hsk/hsv 256/256 supported (E001 support probe).

- **The box is nowhere near a hardware floor, so overhead is the project.** E005: pp8192
  reaches ~3.3 TFLOP/s against a user-reported ~180-190 TFLOPS WMMA fp16 peak, and `GFX` util
  sampled at **9% on all four cards**. tg128 = 35.5 ms/token vs a ~1-2 ms bandwidth floor.
  Consequence: per-kernel work (H4b/QSA) is demoted; cross-GPU sync, host-side stalls and
  placement (E006, F1-F3) are promoted.
- **`size_label` is a lie, structurally**: it is parsed from the repo/file name
  (`gguf-py/gguf/metadata.py:314-328`), so this GGUF reported `A3B` for a ~6 B-active model.
  Derive params yourself (backlog F2).
- **`-lm none` does not disable lazy reads, and no load mode ever prefetches the lazy table**
  - the WILLNEED loop excludes lazy ranges and they are marked `MADV_RANDOM`
  (`src/llama-mmap.cpp:500-510`). Backlog F1.

- **Four mechanisms for the tg gap have now been eliminated by measurement, not argument**:
  weight bandwidth (E006 ratio 0.88 vs 0.25 predicted), cross-GPU collectives (E006, refuted
  by sign), split-mode choice (E006), and graph capture / launch submission (E008, worth only
  ~2.7 ms of a 35.5 ms token). PLE page faults are bounded at ~0.1-4 ms by arithmetic. What is
  left is per-step **host-side graph build/alloc**, which graph replay cannot remove - hence
  E009/E010/E011.

- **`-ot per_layer_token_embd=CPU` has never applied** - lazy-read tensors ignore tensor
  overrides (`W llama_model_loader: tensor overrides do not apply to lazy-read tensors`), and
  lazy mode itself forces the CPU buffer type (`src/llama-model-loader.cpp:1080-1086`). The
  placement is real; the mechanism was not what F3 claimed. See E011.

- **A host-side graph pass costs ~20-22 ms on this box** (E010): disabling graph reuse
  dropped tg 28.20 -> 17.36 while every pp row stayed within noise, and pp512's whole test
  gained 19.6 ms for its single build. Two independent measurements bounding the same number
  is why it is trusted. Per-step cost is invisible in prefill (amortised over ~512 tokens) and
  dominant in decode - so **pp numbers must never be used to reason about decode**.
- **Six mechanisms for the ~28 ms/step are now excluded**: weight bandwidth, cross-GPU
  collectives, split mode, graph capture, scheduler op-offload (E015), same-size graph realloc
  (E009). Reuse works. What remains is per-step host work that flags cannot reach -> E016 is a
  host profile, not another A/B.

## Status legend

`planned` (id reserved, nothing run) | `running` | `done` | `blocked` (external
condition missing) | `dead-end` (hypothesis rejected, kept so nobody retests blind)
