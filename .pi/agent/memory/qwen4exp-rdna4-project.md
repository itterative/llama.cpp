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
| T2 | `bench-4x-r9700-32g` (separate, unreachable) | real `qwen4exp`, end-to-end pp/tg, multi-GPU | nothing; user runs it by hand |

Both boxes are **gfx1201** (user-confirmed for the bench box), so they share an ISA target:
4x Radeon AI PRO R9700 @ 32 GB = 128 GB VRAM, and the dev build's gfx1201-only binary is
valid there. Sharing a target does not make numbers comparable across the boxes - ROCm
version and loader state still have to match.

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
3. **ROCm here is 7.1.1** (hw profile v2), Fedora 44, kernel 7.2.5-200 - the same kernel as the
   bench box. This **supersedes** an earlier note here claiming 6.4.4 / `/opt/rocm-6.4.0`, which
   was true until 2026-09-17 17:47: the upgrade deleted the SONAMEs our build links
   (`libamdhip64.so.6`, `librocblas.so.4`, `libhipblas.so.2`), so `build/` is unloadable and every
   v1 dev number (E001, E002) belongs to the old stack. Prefix is unchanged
   (`/usr/lib64/rocm`), so the cmake line is unchanged. See B5.
4. **`test-backend-ops` reportedly crashes on AMD GPUs** (user statement, not yet
   reproduced on this branch). Until triaged it cannot serve as the correctness gate.
5. Local GPU is `gfx1201`, RX 9070, 56 CU, 16304 MiB, `VMM: no`, wave 32, and it drives
   the display - it is not a quiet measurement device.

## Status (as of 2026-09-17)

Scaffolding built: PROTOCOL, INDEX, E001 (harness viability, `blocked` by the traps
above), E002 (synthetic baseline, `planned`), E003 (QSA tax, `planned`), hw profiles
(both boxes), backlog B0-B3 / P1-P5 / H1-H8 + H4a/H4b + L1-L3, and
`plans/model-shape.md` (real dims vs the synthetic dummy). Memories written:
`qwen4exp-arch`, `experiment-protocol`, and `rdna4-rocm-build` (from the `scout-1` survey
merged with my own re-reads, which corrected three claims of mine: the HC op set is
`_PRE`(gated) + `_POST`(null-comb) only; HIP has **no** shape gate on HC - that was
Metal/Vulkan; and the WMMA flash-attn branch is `fattn.cu:667`, not `:655` which is
CDNA-only).

Headline finding, verified: **qwen4exp builds the QSA indexer and a per-layer mask
rebuild, then calls attention with `n_kv_max = 0`** - and the compaction that would make
it pay off aborts under `GGML_USE_HIP` (`ggml/src/ggml-cuda/fattn.cu:93-97`). See
`qwen4exp-arch` and E003.

## Model ground truth (from HF config.json, text only - no weights)

`qwen4exp` is a **180 B-parameter MoE** (179,999,981,424 params, 360 GB bf16), 48 layers,
hidden 2560, head_dim **256**, 24 Q heads / 2 KV heads, **512 experts with 10 active**,
36 linear-attention + **12 full-attention** layers (interval 4), 262,144 context, VLM (has
a vision tower). Full table in `experiments/plans/model-shape.md`.

How the user actually runs it: **Q4_K_M** weights spread over the 4 GPUs (~80 GB), the
**PLE n-gram table at Q5 in system RAM**, and `mmproj` in RAM to keep full context.
Capacity is therefore not the open question; **bottleneck identity is**.

The PLE table is **~51 B params = 28% of the model, serving exactly one layer**, gathered
via host-side n-gram hashing (ggml has no int64/xor). That single fact reordered the
backlog: L1 (is that table resident or `--lazy-mode` page-faulting?), L2 (host-side gather
+ graph split per ubatch), L3 (10-of-512 expert routing) now sit ahead of the HC/QSA kernel
threads as first-guess `tg` bottlenecks.

Run ledger so far: **E001 done** (B1 closed - `test-backend-ops` passes on gfx1201; dummy
models generate; the qwen4exp graph runs on `ROCm0`; **`test-fusion` is Metal-only so it
cannot run here**), **E002 dead-end** (a 19 MB synthetic model measures harness overhead, not
the bottleneck), **E005 done** - first real numbers, from the bench box on the user's fork
(`pp8192` 547 t/s, `tg128` 28.2 t/s, **9% GPU util**), **E006 planned** (flag-only test of
whether `-sm tensor` collectives are what is costing `tg`), E003/E004 still instrument-less.

**E005 inverted the plan, and E006 narrowed it again.** Both pp and tg are one to two orders
of magnitude off the hardware floor (pp ~3.3 TFLOP/s vs ~180-190 TFLOPS WMMA peak; tg
35.5 ms/token vs a ~1-3 ms bandwidth floor), so this box is *waiting*, not working, and
per-kernel work - including the QSA compaction port (H4b) I was keen on - is demoted. My own
collective-latency explanation for `tg` was **refuted by sign** in E006: `-sm layer` does
essentially no cross-GPU reduces and is *slower*, and its 0.88 ratio vs a predicted 0.25 also
kills bandwidth-bound. What survives is that the `tg` cost is **serial, host-side, and
independent of split mode** - currently blamed on `-ot per_layer_token_embd=CPU` forcing a
CPU gather + H2D + graph split at layer 1 every step, possibly defeating HIP graph capture
entirely.

Immediate next steps: **E014** (`-lzm off`, keeping `-ot ...=CPU` which now finally applies) - a
resident table with no demand paging, and the cleanest remaining cut at the fixed per-step cost;
plus **E008b** (`free -h` / `vmstat` / `iostat` during decode, and system RAM), which E014 depends
on. **E011 is the headline so far: ~90% of a decode step is fixed per-step cost** (~28 ms) no
matter how many tokens it carries, and 40 k depth costs only ~10% of tg - which also bounds what
the QSA port could ever return on decode. Remaining graph-side probes: E009
(`GGML_SCHED_DEBUG_REALLOC=1`), E010 (`LLAMA_GRAPH_REUSE_DISABLE=1`). **B4 - pull in the fork's
MMQ fixes, custom AllReduce, and qwen4exp tensor-split enablement** - is still the top code
action, blocked only on getting the diff here. B0 (`CMAKE_BUILD_RPATH`) stays parked at the
user's choice.

Durable conclusions from E005/E006/E008: **tensor split stays** (it wins decode, as the user
expects; layer *usually* wins pp via inter-layer pipeline overlap, which our run did not see -
still unexplained); **E007 is dead** (PLE is mirrored under `-sm tensor`, so ~30 GB becomes
~30 GB per card, `src/llama-model.cpp:513-515`); **graph capture is not the problem** (E008);
and the **QSA compaction port (H4b) is demoted** as a pp-only optimisation on a subsystem not
yet shown to be the bottleneck.

Two facts that keep paying off: **`GGML_CUDA_DEVICES` exposes virtual devices, so multi-GPU
behaviour is testable on this 16 GB box** (N3), and **`mma_f16` flash attention is used for
prompt processing but not decode on gfx1201** (P4).

## Related memories

- `qwen4exp-arch` - arch/code map of the model itself
- `rdna4-rocm-build` - build recipe and RDNA4-specific backend behaviour
- `experiment-protocol` - how experiments are designed and recorded
- existing: `llama-cpp-quirks` (tool-calling/grammar, unrelated to this project),
  `searching-code` (use `ast-outline` before grep/Read)
