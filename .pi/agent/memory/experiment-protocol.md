---
name: experiment-protocol
description: How qwen4exp/RDNA4 experiments are designed, measured, and recorded on this branch. Read before running any benchmark or writing any experiment record.
category: workflow
priority: 5
keep_updated: true
---

# Experiment methodology (summary)

Full version: `.pi/agent/memory/experiments/PROTOCOL.md`. Ledger: `experiments/INDEX.md`.
Records: `experiments/runs/E<nnn>-<slug>.md`. Raw: `experiments/results/`.

The point of all of this is that perf numbers on 4 consumer RDNA4 cards are noisy and
easy to fool yourself with, and the failure mode is believing a delta that came from
thermals, a stale shared library, or a device filter that matched nothing.

## Non-negotiables

1. **Hypothesis, deciding metric, and expected effect size written before running.**
   Fill the top of the record first. A number that only supports a conclusion you
   already reached is not evidence.
2. **Baseline and variant measured in the same session, interleaved A/B/A/B.** Never
   compare across sessions, machines, ROCm versions, or commits.
3. **Correctness gate before any perf claim.** Touched ops must be shown correct, plus a
   fixed-prompt logprob diff (`scripts/compare-logprobs.py`) with the max/mean absolute
   delta recorded. A faster wrong kernel is worth nothing. If the gate tool is broken,
   say in the record which substitute was used - never silently skip it.
4. **Median of >= 3 repeats with min-max spread.** Spread > 5% means contaminated:
   find the cause, rerun. Single runs are never results.
5. **One hypothesis per `E<nnn>`.** Refinements get a new id with `parent:` set.
6. **Negative results stay in.** `dead-end` rows in `INDEX.md` are what stop repeat
   work.
7. **Raw output is committed with the record**, unedited. No hand-transcribed numbers.
8. **Durable conclusions get promoted to a topic memory**, linking the run id. The log
   is the audit trail; the memory is what survives context loss.
9. **Comparability breaks get a row in `INDEX.md`.** ROCm upgrade, driver change, build
   mode change: bump `profile: vN` in the relevant `hw/*.md`.

## Evidence tiers

- **T1** - dev box, minutes: op correctness, per-op throughput, fusion counts, graph
  splits, synthetic `qwen4exp` model. Authoritative for *does the mechanism work*.
- **T2** - 4-GPU box, hours, manual: real weights, end-to-end t/s, VRAM, scaling.
  Authoritative for *does it matter*.
- A T2-only number with no T1 anchor is tagged `unmechanised` in the record: a data
  point, not a finding.

## Metrics, always separately

`pp<N>` t/s is compute-bound (fusion, mma, batch shape). `tg<N>` t/s is bandwidth-bound
(quant dequant, MoE routing). Plus peak VRAM per device and graph-split count. Do not
average them, do not trade one for the other silently - a change that improves `pp`
while hurting `tg` is a decision, not a result.

## An effect under ~5% needs paired, alternating runs

Measured noise on the bench box, same binary and flags, `-r 3`: 0.7-1.0% on tg at d4096-d40960 and
**2.6% at d131072**, from an A/B that turned out to run the identical binary in both arms. So `-r 3`
sequentially cannot resolve a claim below about 5%, and this bit twice: E055's first 4-card A/B, and the
first reading of the one-shot allreduce (logged as +7.2%/+8.1% from two separate `-r 3` invocations;
paired and alternating at `-r 10` it is +2.5% at depth and zero at 4k).

Required before an effect is written down as a result: both arms **in one session, alternating**
(`for pass in 1 2; do for arm in A B; do ...`), `-r >= 10`, one depth per invocation if the harness mixes
phases. And prefer a mechanism check over more reps where possible - a measured cost that predicts the
gain by independent arithmetic (us/call x calls/step / step time) is stronger evidence than a t/s delta,
and it also shows where the model of the change is wrong.

## Before any measurement, on this box

```sh
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin        # ~/.local/lib64 has a stale install
ldd build/bin/llama-bench | grep -E 'lib(ggml|llama)\.so\.0 '
```

Device filter is exactly `ROCm0`, and a filter that matches nothing *passes silently*. Same trap for
feature gates: an equal result between two arms means nothing until something proves the new path
ran - print or grep one line only the live path emits. For the qsa pool that line is
`qsa pool: mode = 1`, and it is `LLAMA_LOG_DEBUG`: it needs `-v`/`--verbose`, so without it an arm that
never pooled is indistinguishable from one that did. (`mode = 2`/REBUILD is gone since E051; only
NONE=0 and CACHED=1 exist.) Two separate times this produced a false "identical, so it is safe".

**The standard golden gate is vacuous for anything gated on a single sequence.** `llama-perplexity`
derives `n_seq = max(1, n_batch/n_ctx)`, so the golden command (`-c 512 -b 2048`) runs **4 sequences per
batch** and `qsa_pool_get` answers NONE for all of them. E044/E049/E051/E052 all cite `263113.6984` as
the pool's gate; in E056 both of those arms measured 0 mode lines, i.e. pool-off vs pool-off. Gates that
do engage it: the sparse corpus at `-c 8192 -b 2048` (n_seq=1), `-b 256 -c 2048`, the rollback harness,
and a greedy `llama-cli -c 4096 -n 2500 -st` text diff. `-c` also has to satisfy
`2*n_ctx <= corpus tokens`, so the 3428-token golden corpus cannot run below `-c 1714`.

## Region profiling inside the process (`02d963eb0`)

`ggml_prof_region("name")` from `ggml/include/ggml-prof.h` brackets a host phase. With
`GGML_PROF_REGIONS=1`, ggml-base counts calls / total / max and prints a table at exit. A backend may
register a `ggml_prof_sink` that mirrors each region into its own profiler; the HIP backend does that
by loading the roctx marker library by name (`ggml/src/ggml-cuda/prof-roctx.cu`), so a rocprofv3 trace
groups kernel rows by phase. **Annotations only - nothing here collects data, and collection stays in
the external tool.** The dev box has no roctx SDK and no rocprofv3, so `regions on: counters` is the
correct output here, not a failure.

Live regions: `graph:build`, `graph:alloc`, `graph:set_inputs`, `graph:compute` (llama-context),
`meta:subgraph`, `meta:allreduce` (tensor/row split dispatch), `spec:draft_decode` (draft-mtp),
`ckpt:save_tgt`, `ckpt:load_tgt` (the rollback door), `phase:decode` (`0f641cc1b`).

### Capturing decode only

`GGML_PROF_DECODE=<max n_tokens counted as decode>` opens a capture window (`roctxProfilerResume` /
`Pause`) on the first batch at or below that width and closes it on the next wider batch or at exit, so a
whole-turn trace carries decode kernels and nothing from the prompt pass. It also brackets a
`phase:decode` region, which reports per-step decode wall time at exit with no profiler attached at all -
often enough on its own, and the thing to reach for before fighting rocprofv3.

- plain decode: `GGML_PROF_DECODE=1`
- **speculative decode: `n_draft + 1`, not 1.** A verify pass wider than the limit closes the window and
  the next draft step reopens it, so the trace arrives chopped into slivers. Set the limit above the
  verify width to keep one contiguous region per turn.
- the draft context's own single-token steps go through the same hook, so the region count is draft steps
  plus target steps, not tokens. that is usually what is wanted for H18, but it is not a token count.
- unverified from the dev box: whether `--selected-regions` is the flag that consumes the resume/pause
  pair (no rocprofv3 or rocprofiler-register installed here, so the claim in `ggml-prof.h` came from docs
  and not from `--help`). check `rocprofv3 --help | grep -i -A3 region` on the bench box. if their build
  lacks selected-region support, the roctx ranges still arrive as marker rows to slice on afterwards, and
  the counters work either way - the window is an optimization on top, not the mechanism.
  `[x]` confirmed on the bench box: `--selected-regions` is real there, and its own warning names
  `roctxProfilerResume(0)`/`roctxProfilerPause`, i.e. exactly the pair `ggml_prof_window` calls.

### Two traps in the roctx sink itself (`5acbd6377`)

- **resolve at first use, never at load.** `prof-roctx.cu` used to dlopen the shim from a static
  initialiser, i.e. while an attached profiler is still bringing up its own SDK: under rocprofv3 the
  attempt failed and the null handle was cached forever, so ranges and windows were both silent. The
  pre-`ggml_prof` llama-bench code avoided this by resolving inside the measurement loop. Attempts are now
  bounded (16 on the region path, unbounded when opening a window) so a box without the SDK does not pay a
  failed dlopen per region.
- **`/opt/rocm` is not the only prefix.** The dev box keeps ROCm under `/usr/lib64/rocm` and also has an
  old `/opt/rocm-6.4.0`, so the sink now additionally tries the directory `libamdhip64` was actually
  loaded from (`dladdr` on `hipGetDevice`).

Testing the sink with no profiler attached, on any box that has the roctx lib somewhere:

```sh
LD_PRELOAD=/opt/rocm-6.4.0/lib/librocprofiler-sdk-roctx.so.1 GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1 \
  build/bin/llama-cli -m <small model> -ngl 0 --temp 0 -st -p hi -n 2
```

which prints `[prof] roctx: attached, roctxProfilerResume present ...` instead of `library not found`. The
symbol is present in that SDK build, so a `--selected-regions` run recording nothing means the sink did
not attach, not that the API is missing.

Reading the table, all three of which I got wrong the first time:

- totals are **inclusive** of nested regions. Summing rows is meaningless; subtract children for self
  time. `graph:compute` contains `meta:subgraph`.
- region-ms is a per-thread aggregate, so a run can report far more region-ms than it took in wall
  time. `-r 1` still runs several passes (warmup + estimate), which is why `calls` can be ~4x the
  ubatch count in a bench row. Check `calls` before converting to ms/token.
- `graph:compute` is the *async* entry point: it returns after enqueueing, so its ms is mostly device
  wait only where something inside synchronizes. Do not read it as dispatch cost.
- counters are not locked; overlapping threads can double count. Today's sites are all main-thread.

## Crash and debug loop on this box

- Debug build without losing the HIP objects: `cmake -B build -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g3
  -fno-omit-frame-pointer"` (same for `_CXX_FLAGS_RELEASE`/`_C_FLAGS_RELEASE`). It drops `-DNDEBUG`, so
  `GGML_ASSERT` fires with a message instead of corrupting silently. The variable is cached, so
  re-running plain `cmake` does **not** restore `-O3`: pass it explicitly again.
- llama.cpp log prefixes are `MM.SS.mmm.uuu` (minutes), not days or hours - misreading them invented a
  "15-day uptime" theory that the real timeline contradicted.
- Multi-device emulation is unavailable here: `GGML_CUDA_DEVICES` takes a *count* (a device list is
  rejected) and requesting more devices than exist dies with `invalid device ordinal` on HIP. But
  `-sm tensor` still routes through `ggml-backend-meta.cpp` on one device, which is how a 4-card-only
  bug turned up locally at all.
- Cores: `coredumpctl --all list` (needs `--all`), `coredumpctl --all dump PID > core`, then
  `gdb -batch -ex "frame N" -ex "print ..." <bin> core`. Iterate on the core, not by re-running a
  12 GB model. `print` after `run` in one `gdb -batch` loses the frame.

## Git hygiene for these files

`.pi/` is partially tracked upstream (`.pi/gg/SYSTEM.md`), so `.pi/agent/memory/*` shows
up as untracked in `git status`. Use explicit pathspecs, never `git add -A`, and never
let `.pi/agent/memory/experiments/` into a commit destined for a PR.
