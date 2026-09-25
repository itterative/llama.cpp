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
   bench box. This **supersedes** an earlier note here claiming 6.4.4 / `/opt/rocm-6.4.0`. The
   upgrade deleted the SONAMEs the old build linked (`libamdhip64.so.6`, `librocblas.so.4`,
   `libhipblas.so.2`), so every v1 dev number (E001, E002) belongs to the old stack.
   **B5 is done**: rebuilt on 2026-09-17 with the same cmake line (prefix `/usr/lib64/rocm` is
   unchanged), except that `-DCMAKE_C_COMPILER=clang` in the recipe is wrong here - there is no
   `clang` on PATH, let cmake pick the default.
4. **`test-backend-ops` reportedly crashes on AMD GPUs** (user statement, not yet
   reproduced on this branch). Until triaged it cannot serve as the correctness gate.
5. Local GPU is `gfx1201`, RX 9070, 56 CU, 16304 MiB, `VMM: no`, wave 32, and it drives
   the display - it is not a quiet measurement device.

4. **A HIP device exception dumps `gpucore.<pid>` (~7.3 GB each on this card) into the CWD**, i.e.
   the repo root when tools are run from there. Two crashes were 14.6 GB before anyone looked
   (E020). Check `df` and `ls gpucore.*` after any `HSA_STATUS_ERROR_EXCEPTION`.

## Status (as of 2026-09-17, refreshed after E018)

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

Run ledger: **E001** (B1 closed on the old stack: ops pass, dummy models generate, the graph
runs on `ROCm0`, `test-fusion` is Metal-only), **E002 dead-end** (19 MB synthetic measures harness
overhead), **E005** first real numbers from the bench on the user's fork (`pp8192` 547 t/s,
`tg128` 28.2 t/s, **9% GPU util**), **E006** tensor wins decode / layer wins pp usually, **E007
killed** (PLE is mirrored under `-sm tensor`, ~30 GB/card), **E008** graphs are captured (adding
`GGML_CUDA_DISABLE_GRAPHS=1` costs 7% of tg), **E009** no failed same-size reallocs, **E010**
graph reuse works (~22 ms/step if disabled), **E011 headline** (~90% of a decode step is fixed
per-step cost, ~28 ms, independent of tokens in the step; 40 k depth costs ~10% of tg), **E015**
`-nopo` zero effect, **E017/E018** the local dummy harness (below). Not run: E012 (`-sm row`),
E014 (`-lzm off`), E008b (fault counting), E016 (`perf record -g`), E003/E004 (still
instrument-less).

**E005 inverted the plan and E011 finished the bench phase** (user decision, 2026-09-17): both
pp and tg sit one to two orders of magnitude off the hardware floor (pp ~3.3 TFLOP/s vs ~180-190
TFLOPS WMMA peak; tg 35.5 ms/token vs a ~1-3 ms bandwidth floor), six mechanisms are excluded, one
graph build costs ~20-22 ms (E010) and ~28 ms/step is fixed per-step host cost (E011), so 40 k
depth accounts for only ~10% of tg. That demotes kernel-level work: no per-kernel change can pay
for the fixed cost that dominates it. My own collective-latency explanation for `tg` was **refuted
by sign** in E006, and the "CPU split defeats graph capture" story died in E008.

Durable conclusions from E005/E006/E008: **tensor split stays** (it wins decode, as the user
expects; layer *usually* wins pp via inter-layer pipeline overlap, which our run did not see -
still unexplained); **E007 is dead** (PLE is mirrored under `-sm tensor`, so ~30 GB becomes
~30 GB per card, `src/llama-model.cpp:513-515`); **graph capture is not the problem** (E008);
and the **QSA compaction port (H4b) is demoted** as a pp-only optimisation on a subsystem not
yet shown to be the bottleneck.

Two facts that keep paying off: **`GGML_CUDA_DEVICES` exposes virtual devices, so multi-GPU
behaviour is testable on this 16 GB box** (N3), and **`mma_f16` flash attention is used for
prompt processing but not decode on gfx1201** (P4).


## The dummy harness (what we do locally now)

`experiments/tools/mkq4expdummy.py` writes a shape-faithful qwen4exp GGUF from the real
`config.json` + tokenizer (kept outside the repo, in the session scratchpad) with the real file's
per-tensor types and a seeded payload. Shipped artifact: **`models/q4exp-4l.gguf`** (4 real-dims
layers = lin, lin+PLE, lin, full; 512 experts; 5.5 GB n-gram table row-cut from 320M rows/head to
3.1M; 12.76 GB; ignored by `.gitignore`'s `/models/*`). ~40 s to build; iterate fill/metadata
changes on a `--layers 2 --experts 32 --ple-head-rows 65536` smoke (1.3 GB, seconds) and write the
big one once. Runs on this box as `-ngl 99 -lm none -sm none -fa 1`: **tg128 @ d40960 = 181.89
t/s = 5.47 ms/step**, of which ~4.5 ms is not attributable to GPU math - same class as the bench's
fixed cost, ~1/6 the magnitude, because per-node host work scales with layer count.

Three rules that come from E017/E018 and must not be re-derived:
- **Correctness gate**: `llama-perplexity -m models/q4exp-4l.gguf -f
  experiments/tools/golden-corpus.md` -> `PPL = 262938.7619 +/- 3039.06817`, bit-stable across
  runs. It covers the *prefill* path only, and its contract is "every diff is explained", not "no
  diff" - the QSA compaction port must move it.
- **Noise floor ~1%**: the same file measured 180.3 / 181.8 / 181.9 t/s across invocations, and
  this GPU drives a display. Sub-2% local deltas are not findings.
- **Decode does not care about table volume** (182.1 / 182.7 / 182.4 at 35 / 3.7 / 5.5 GB) but
  **pp4096 gained 6.5% from the small table**, which is the first positive evidence for F1 (lazy
  ranges are never prefetched, get `MADV_RANDOM`, `MAP_POPULATE` deliberately skipped).

## First code change on this branch (E020: `abd3473a8`, `b364ff44e`)

Sparse flash attention runs on RDNA4. `Q4EXP_SPARSE_FA=1` engages above 4102 KV and gives
**pp512 +23.3% at d40960, +58.5% at d163840**, numerically equal to dense-with-mask within
6e-7, with **tg unchanged**. Six blockers had to clear; the ones the code survey missed are in
`experiments/runs/E020-sparse-fa-rdna4.md` (HIP's 64-bit `__ballot_sync`, `may_use_sparse`
whitelisting only DKQ 512/576, and RDNA having no FA device code below 16 tiles, which forces
the 1x16 tiling and a generator change). Open by choice, not oversight: NVIDIA is untested
here, the new instantiation costs CUDA build time and code size, the env default is off, and
the E018 golden corpus (3428 tokens) can never reach the depth gate - use
`tools/sparse-corpus.md` with `-c 8192` for any sparse-path check. H4a stays live: sparse
scans the mask, it does not stop the model from rebuilding it.

## H9 block-key pool state (E044 -> E057, on by default)

`Q4EXP_POOLED` (**on by default since `9111adf2c`**; `=0` is the opt-out) makes the QSA indexer pool each
block key once, when the block completes, instead of re-deriving all of them every step. On 4 cards at
131k it is worth **+32.6% tg** and, since E056, **+2.4% pp** over pool-off (+1.8% over excluding prefill).
Three shape rules came out of the prefill work and are the part to remember:

- **One pooled topology.** REBUILD was collapsed into CACHED with `wm = 0` (`8206d79d8`), and the
  scores read the `set_rows` result, so the write->read dependency is a graph edge and not node order.
- **The reservation must measure the shape the runtime builds.** `sched_reserve` runs through a
  full-cache context with no cells, so `qsa_pool_get` used to answer NONE there and ggml-alloc dropped
  the worst-case budget on the first pooled ubatch, then re-reserved on every ubatch after it (E049's
  ratchet). Fixed in E056 by keying the worst case on `llama_memory_hybrid_idx_context::is_update` -
  the context kind, not the cell state - and by pooling one masked row when a run is shorter than a
  block, which is what the tools' 1-token warmup decodes. 0 re-reserves, +0.08 MiB of reservation.
- **A re-reserve costs ~25 ms here and ~300 ms on 4 cards**, mostly the device drain plus the lost
  host/device overlap, so on the bench box it is worth ~110 ms per prefill ubatch (run8: pp8192 1781
  pooled-prefill vs 2230 with prefill excluded). Cheap to detect: `GGML_PROF_REGIONS=1` and look for
  `sched:realloc_size` / `sched:realloc_buft`.

`Q4EXP_POOLED_NO_PREFILL` is **deleted** (`9111adf2c`, with the default flip): with the reservation fixed,
pooling prefill measures slightly positive and is cheaper on the host, because the pooled variant never
creates `blk_pos` (I32 `[4*n_blocks*n_stream]`, 557 KB per ubatch at 131k), which is where
`graph:set_inputs` spends 125.9 ms pool-off and 90.8 ms pooled. The pooled reservation is also 34.8 MiB
*smaller* at that context.

### What the pool cannot cover, and the crash it hid (E057)

- **No session containing an mrope image ever pools**, in the target or in the draft. Pooling and the
  block-bias fast path both require the used cells to be a dense 1:1 run of positions
  (`llama-memory-hybrid-idx.cpp:379-388`), and an image pins `nx*ny` cells to one `t`. So **+32.6% tg
  is text-only**, and no gate we own could show it: golden, sparse corpus, greedy decode and rollback
  all feed dense text, where position, cell and rank numbering coincide. H20 closes it (pool in rank
  space, which is append-stable for mrope exactly like position space).
- **The same requirement was the field crash.** A user conversation with 3 images aborted on
  `qsa: cell position runs past the cell window` in `ctx_dft`: `draft_mtp::process` returns early on
  embedding batches (`common/speculative.cpp:1491-1493`) so image cells never enter the draft cache,
  while the positions it copies keep the image's `max(nx, ny)` advance - positions run past the window
  that `get_n_kv()` sizes from the *cells*, and with `r = 4` that window is `ceil256(cell_p1)`, so the
  slack a conversation has to beat is under 256 cells, not the context size. Fixed by `ebe30e1fd`: rank
  the cells whenever the position line does not step once per cell. **It was never the pool**
  (`blk_bias` comes from the mask shape and causality; both pool arms abort identically) and it predates
  H9. E057 has the prediction-then-observation table and the two further states the fix exposed
  (`n_bid` left dirty by a bailed `try_contiguous`; `new_rows`/`new_cells` never named when the pool was
  promised at build and the run broke after it, which sent the indexer gather off the cache into a GPU
  fault).
- What still builds the historic graph, and so still churns against a pooled reservation: more than one
  sequence in the cells (needs `--kv-unified` with >1 slot), **any run with an image in it**, a non-dense
  run from an interior `seq_rm`, or a run whose endpoints satisfy the pool's test while its interior does
  not. Safe since E057, but the last one rebuilds the graph per 256-cell bucket (H21), and a pinned or
  gapped run counted 19-21 `sched:realloc_size` per 7.5k cells against 0 for dense (H22, unresolved).

Two things the default flip changes that no numeric gate catches: the pool's **354 MiB/card at ctx
245760** is now paid by every qwen4exp run (H15/P3's f16 storage is the fix, and it just got more
relevant), and a plain `llama-bench` on this branch is no longer comparable to a pre-`9111adf2c` number
unless that number set `Q4EXP_POOLED` explicitly - same trap as E055's `855a65544`.

## Next steps

Top code action is the comms thread (`plans/decode-comms-plan.md`): the one-shot allreduce is in and
measured at +2.5% tg at depth, with three loose ends left. H9 is measured end to end and on by default;
what is left on it is vision - H20 (pool nothing while an image is in the run), H21 (the pool promises on
an endpoint test the fast path does not honour) and H22 (does a vision turn pay H19's ratchet; needs a
bench reading, because the dev box counts ~20 re-reserves per 7.5k cells while wall time moves 1%). The
VRAM tax the default now makes everyone pay is still open too. H19 is the non-pool
reservation ratchet. **E058 is armed and needs a bench-box run**: what the n-gram fetch costs a decode
step under `-lzm on-direct`, and what share of its rows are cold. `126b7a43b` added the instrumentation
(prof regions inside `graph:set_inputs`, plus `/proc/self/io` counters bucketed decode vs prefill), and
the answer decides between a one-line worker-divisor fix and a GPU row cache. B4 (the fork diff) and N4
were dropped in the backlog sweep; `test-backend-ops`
still needs a 7.1.1 re-baseline before it can serve as a correctness gate.

Gate for any QSA block-numbering, pool or mrope change: `experiments/tools/qsa-posgap-harness.cpp`.
It feeds `seqN` / `pinN` / `gapN` scripts to one plain context and fingerprints the final logits, so it
catches the index-space class of bug that every existing gate is blind to. Run both pool arms.


## Related memories

- `qwen4exp-arch` - arch/code map of the model itself
- `rdna4-rocm-build` - build recipe and RDNA4-specific backend behaviour
- `experiment-protocol` - how experiments are designed and recorded
- existing: `llama-cpp-quirks` (tool-calling/grammar, unrelated to this project),
  `searching-code` (use `ast-outline` before grep/Read)
