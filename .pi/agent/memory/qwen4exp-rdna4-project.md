---
name: qwen4exp-rdna4-project
description: Active local-branch project - optimizing the qwen4exp arch for RDNA4/ROCm; machine roles, artifact map, measurement traps, current status. Read first in this repo.
category: project
priority: 5
keep_updated: true
---

# qwen4exp on RDNA4 - project state

Worktree `/home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp`, branch
`experiments/qwen4exp-rdna4`. Local-only branch: **never push it, never PR from it
without a fresh discussion with the user.** Main checkout lives at
`/home/sd/Repos/llama.cpp` (branch `master`).

Goal: performance work on the `qwen4exp` arch (HF `Qwen/Qwen3.8-Flash-Next`) using the
**HIP/ROCm** backend on RDNA4. Not Vulkan. The user's phrase for the family is
"qwen4exp" (they first said "qwen4flash" and corrected it).

## Two-machine reality

| role | machine | can do | cannot do |
|---|---|---|---|
| T1 | `dev-rx9070-16g` (this box) | build, op-level, synthetic models, per-op timing | real weights (needs > 16 GB) |
| T2 | `bench-4xr9700` (separate, unreachable) | real `qwen4exp`, end-to-end pp/tg, multi-GPU | nothing; user runs it by hand |

The user **cannot download the model at all** (insufficient SSD). So synthetic
`qwen4exp` GGUFs via `test-llama-archs` are the primary local substrate. Do not
propose `huggingface-cli download` / conversion runs.

## Artifacts

Everything under `.pi/agent/memory/experiments/` (log tree) + the topic memories listed
at the end of this file. Read `experiments/PROTOCOL.md` before running or recording
anything - it defines the tiers, the naming, and the rules.

## Traps that already produced bogus results

1. **Loader shadowing.** `LD_LIBRARY_PATH` includes `/home/sd/lib` and
   `/home/sd/.local/lib64`, and `~/.local/lib64` holds a **stale Sep 15 llama.cpp
   install**. The build is `BUILD_SHARED_LIBS=ON` with no rpath, so an unpinned binary in
   `build/bin/` loads the *old* libggml/libllama. Symptoms seen: `symbol lookup error:
   undefined symbol: ggml_dsv4_hc_pre_gated`, and `test-llama-archs` emitting 0 files
   with `key llama.attention.causal has wrong type f32 but expected type bool`.
   Always `export LD_LIBRARY_PATH=$PWD/build/bin` and confirm with `ldd`.
2. **Device name is `ROCm0`.** Not `CUDA0`, not `HIP0`. `test-backend-ops -b` and
   `test-fusion --device` take it verbatim. A non-matching filter is **silent**: it
   prints "N/N backends passed / OK" having tested nothing.
3. **ROCm here is 6.4.4**, not the 7.2 the user believed. Fedora-packaged, `/opt/rocm-6.4.0`,
   hipcc `19-14.rocm6.4.2.fc43`, `offload-arch` absent, `AMDGPU_TARGETS` unset (so the
   binary is gfx1201-only). The user intends to upgrade - that is a comparability break
   when it happens.
4. **`test-backend-ops` reportedly crashes on AMD GPUs** (user statement, not yet
   reproduced on this branch). Until triaged it cannot serve as the correctness gate.
5. Local GPU is `gfx1201`, RX 9070, 56 CU, 16304 MiB, `VMM: no`, wave 32, and it drives
   the display - it is not a quiet measurement device.

## Status (as of 2026-09-17)

Scaffolding built: PROTOCOL, INDEX, E001 (harness viability, `blocked` by the traps
above), E002 (synthetic baseline, `planned`), E003 (QSA tax, `planned`), hw profiles,
backlog B0-B3 / P1-P5 / H1-H8 + H4a/H4b. Memories written: `qwen4exp-arch`,
`experiment-protocol`. Still pending: `rdna4-rocm-build` (cmake wiring, env knobs, which
FA family HIP uses on gfx1201) - waiting on read-only survey `scout-1`.

Headline finding, verified: **qwen4exp builds the QSA indexer and a per-layer mask
rebuild, then calls attention with `n_kv_max = 0`** - and the compaction that would make
it pay off aborts under `GGML_USE_HIP` (`ggml/src/ggml-cuda/fattn.cu:93-97`). See
`qwen4exp-arch` and E003.

Immediate next steps, in order: resolve B0-B1 (unshadowable build, `test-backend-ops`
crash triage), then rerun E001 with the pin, then E002 to get the first real baseline.
Bench-box characterisation (B2) is waiting on the user to run the paste-block in
`experiments/hw/bench-4xr9700.md`.

Strongest verified lead so far: **`-sm tensor` is not supported for qwen4exp** -
`llm_arch_supports_sm_tensor` returns false for `LLM_ARCH_QWEN4EXP`
(`src/llama-arch.cpp:1161`, upstream comment `// TODO: fix test-llama-archs`), which
constrains how the 4-GPU box can be balanced at all.

## Related memories

- `qwen4exp-arch` - arch/code map of the model itself
- `rdna4-rocm-build` - build recipe and RDNA4-specific backend behaviour
- `experiment-protocol` - how experiments are designed and recorded
- existing: `llama-cpp-quirks` (tool-calling/grammar, unrelated to this project),
  `searching-code` (use `ast-outline` before grep/Read)
