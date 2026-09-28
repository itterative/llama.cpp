---
name: env-knobs
description: Index of every environment variable and compile-time knob this branch adds to llama.cpp - default, opt-out, what it does, where it is read, and the record that measured it. Read before setting an env var or wondering whether one exists.
category: project
priority: 3
keep_updated: true
---

# Env knobs this branch adds

Anchor: every change on `experiments/qwen4exp-rdna4` is after upstream `ebbb18522`. To regenerate the
list from the code (it is the authority, this file is the map):

```sh
git diff --name-only ebbb18522...HEAD | grep -E '\.(cu|cuh|cpp|h)$' > /tmp/t.md
grep -rhoE '(getenv|std::getenv|ggml_cuda_ar_env_u64)\("[A-Z0-9_]+"' $(cat /tmp/t.md) | grep -oE '"[A-Z0-9_]+"' | sort -u
```

Upstream's own variables (`GGML_CUDA_DISABLE_FUSION`, `GGML_SCHED_DEBUG`, `LLAMA_TRACE`, ...) also show
up in these files; everything below was added by the branch. Two names appear only in old records and
**do not exist in the code**: `Q4EXP_NO_INDEXER`, `Q4EXP_FA_DEBUG` - do not pass them.

## The standard config - all default-on since 2026-09-27

A bare `llama-bench` / `llama-server` run needs no env. Defaults from `6344fd41b` (ggml side),
`f383ef73e` (`Q4EXP_SPARSE_FA`), plus the earlier flips `9111adf2c` (pool), `855a65544` (small_k),
`3f1138bb3` (lazy prefetch).

| variable | default | opt-out | what it does | read in | record |
| --- | --- | --- | --- | --- | --- |
| `GGML_CUDA_P2P` | on | `=0` | explicit peer-access enable for every device pair, plus the VMM peer grant | `ggml-cuda.cu` | H16 |
| `GGML_CUDA_ALLREDUCE` | `internal` | `=nccl`, `=none` | which comm backend the meta backend picks; `internal` is this branch's own allreduce | `ggml-cuda.cu` `ggml_backend_cuda_comm_init` | E045, E046 |
| `GGML_CUDA_AR_DIRECT_BF16` | `nccl` | `=off` | compress large F32 reductions to a bf16 wire, NCCL's element-count heuristic; `<bytes>` sets the threshold. **Changes numerics** | `allreduce-p2p.cu` | H16 |
| `GGML_FATTN_RDNA_RTILE` | on | `=0` | use the rtile RDNA FA decode kernel where it can engage | `fattn-rdna.cu` | E034, E047 |
| `Q4EXP_SPARSE_FA` | on | `=0` | sparse FA path, i.e. pass `n_kv_max = top_k->ne[0]` to `build_attn_mha` | `qwen4exp.cpp` | E022, H13 |
| `Q4EXP_POOLED` | on | `=0` | pool indexer block keys at write time instead of re-deriving them per step | `llama-memory-hybrid-idx.cpp` | E044-E057 |
| `GGML_CUDA_MMVQ_RDNA4_SMALL_K` | on | `=0` | RDNA4 mmvq block shape (worth 5.5-6.6% tg on 4 cards) | `mmvq.cu` | E055 |
| `LLAMA_LAZY_PREFETCH` | on | `=0` | gather all distinct rows before waiting on any (the n-gram fetch, 5.4% -> 2.3% of the wall) | `llama-lazy-reader.cpp` | E058 |

For A/B arms: an "off" arm that *omits* the variable is now the ON arm. Say `=0` (or the documented
alternative value) explicitly.

## Measurement instruments - off unless set

| variable | default | values | what it does | record |
| --- | --- | --- | --- | --- |
| `GGML_CUDA_MMVF_BLOCK_SIZE` | `0` = heuristic | rows per block | overrides the mmvf block size; blanket overrides lost end to end | E060 (negative) |
| `GGML_CUDA_AR_ONESHOT_PROBE` | `0` = off | `<iterations>` | one-time startup self-test of the one-shot path over the real links, and logs its latency; runs once, before model work | E061, E063, E064 |
| `GGML_CUDA_AR_DIRECT_ALGO` | `auto` | `auto`/`oneshot`/`butterfly`/`ring`/`bde` | force an allreduce algorithm instead of the size-based pick | comms plan |
| `GGML_CUDA_AR_PIPELINE` | `auto` | `host_staged`/`direct` | force the copy-based or direct-P2P pipeline | comms plan |
| `Q4EXP_CELL_SEL` | off | set = old path | keeps the old cell-expand selection for A/B against the current one | H13 |
| `GGML_FATTN_RTILE_PB` | unset = heuristic | parallel blocks | override the rtile minimum parallel-blocks gate | E047 |
| `LLAMA_LAZY_WORKERS` | `0` = heuristic | threads | reader threads per gather (default is one per 32 rows, which leaves a decode gather on one thread) | E058 |
| `GGML_PROF_REGIONS` | off | set = on | region counters (`[prof]` tables); also switches the roctx sink on | E053, E059 |
| `GGML_PROF_DECODE` | `1` | `<tokens>` | width at or below which a batch counts as decode: sets the phase label, and setting it also opens the tg window (legacy spelling of `GGML_PROF_WINDOW=tg`) | E059, E061 |
| `GGML_PROF_WINDOW` | unset = no window | `pp`/`prefill`, `tg`/`decode`/`1`, `both`/`all`, `off`/`0` | which phase the roctx capture window covers, i.e. what a `rocprofv3 --selected-regions` run records; `pp` is new, `both` is one window spanning the run | dev box, ROCm 10, 2026-09-28 |
| `GGML_CUDA_NCCL_RETRY` | retry ON | set = off | disable the NCCL failure retry | - |
| `GGML_FATTN_DEBUG` | off | int | FA selection debug | - |
| `GGML_CUDA_AR_ONESHOT_DEBUG` | off | set = on | dumps the one-shot pointers/inbox state | - |
| `GGML_ALLOC_DEBUG_REALLOC` | off | set = on | ggml-alloc names the tensor that fails the reservation fit test (shape, need, recorded size) and the structural trips, dumps the reserve and runtime node lists once each, and the scheduler prints the dependency maps and the dep nodes it emits; needs `-v` | E069, E071, E072, E073 |

Note on the tracer: `GGML_PROF_REGIONS=1` alone is host-side only and cheap; adding `rocprofv3` inverts
some cross-arm results (see `experiment-protocol`, "a tracer can invert a relative result"). A kernel
trace of either phase needs `GGML_PROF_WINDOW`, the phase label is unaffected by it, and with no window
set the tracer records nothing at all - `llama-bench`'s own implicit tg window was removed in
`cc6be7d35`, so a pp and a tg window in one run no longer bleed into each other.

The region tables list **only regions that were hit**: an absent row means zero calls, not a missing
instrument. That is what makes an absence readable - the bench box's missing `sched:realloc*` rows are a
true zero (E074), verified against the same pre-fix commit printing them on the dev box (E075).

## Internal allreduce tuning - leave alone unless debugging it

All in `allreduce-p2p.cu` / `allreduce-host.cu`, all with working defaults:
`GGML_CUDA_AR_DIRECT_TMP_BYTES` (16 MiB, the inbox reservation, 4x per device),
`GGML_CUDA_AR_DIRECT_ONESHOT_BYTES` (256 KiB, above it the one-shot defers to auto's pick),
`GGML_CUDA_AR_DIRECT_AUTO_RING_BYTES` (1 MiB, the butterfly/ring crossover),
`GGML_CUDA_AR_DIRECT_RING_ORDER` (`auto` = measure, also `identity`/...),
`GGML_CUDA_AR_BF16_THRESHOLD` (1, host-staged path's bf16 wire; `0` disables),
`GGML_CUDA_AR_COPY_THRESHOLD` (1 MiB), `GGML_CUDA_AR_COPY_CHUNK_BYTES` (0 = heuristic, 256 KB floor).

## Compile-time knobs (no env)

| knob | default | where | record |
| --- | --- | --- | --- |
| `MMVF_K_UNROLL` | 4 on gfx12, 1 elsewhere | `mmvf.cu` | E061: mmvf device time -21.8%, +1.9% tg untraced |
| mmvq narrow-K block table | hardcoded per device table | `mmvq.cu` (`c_narrow_promoted`) | E059: why a non-RDNA4 device in an RDNA4 build still gets RDNA4 shapes |

For the standard bench environment as a script variable, see `bench-finish-bundle.md` (its REF block
names every one of these explicitly, so it stays valid if a default moves again).