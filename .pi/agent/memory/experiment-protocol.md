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

## Before any measurement, on this box

```sh
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin        # ~/.local/lib64 has a stale install
ldd build/bin/llama-bench | grep -E 'lib(ggml|llama)\.so\.0 '
```

Device filter is exactly `ROCm0`, and a filter that matches nothing *passes silently*. Same trap for
feature gates: an equal result between two arms means nothing until something proves the new path
ran - print or grep one line only the live path emits (e.g. `block key pool = 1`, `qsa pool: mode = 2`).
Two separate times this produced a false "identical, so it is safe".

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
