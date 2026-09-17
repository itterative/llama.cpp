# INDEX - qwen4exp / RDNA4 experiments

Append-only ledger. One row per experiment. Newest at the bottom (ids never reuse).
Detail lives in `runs/`; this table is for scanning and for answering "did we already
try that?".

| id | date | tier | machine | hypothesis (one line) | headline result | verdict | record |
|---|---|---|---|---|---|---|---|
| E001 | 2026-09-17 | T1 | dev-rx9070-16g | the T1 harness (dummy models + fusion counts + bench) runs on ROCm0 | harness not yet clean: loader trap + `test-llama-archs` produced nothing | blocked | [runs/E001-t1-harness-viability.md](runs/E001-t1-harness-viability.md) |
| E002 | - | T1 | dev-rx9070-16g | a synthetic qwen4exp model gives a stable, reproducible pp/tg baseline | - | planned | [runs/E002-synthetic-baseline.md](runs/E002-synthetic-baseline.md) |
| E003 | - | T1 | dev-rx9070-16g | qwen4exp pays a context-scaling QSA tax for sparsity it cannot collect on ROCm | - | planned | [runs/E003-qsa-tax.md](runs/E003-qsa-tax.md) |

## Comparability breaks

Any row here means numbers on either side of it are not valid A/B partners.

| date | machine | change | invalidated |
|---|---|---|---|
| - | - | (none yet) | - |

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
weights) to fill.

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
- `-sm tensor` is not supported for this arch (`src/llama-arch.cpp:1161`), so 4-GPU runs
  are limited to layer split. Drives H1.
- The dev box's `~/.local/lib64` shadows the working tree for every dynamic binary, and it
  already produced two false failures. Drives B0.

## Status legend

`planned` (id reserved, nothing run) | `running` | `done` | `blocked` (external
condition missing) | `dead-end` (hypothesis rejected, kept so nobody retests blind)
