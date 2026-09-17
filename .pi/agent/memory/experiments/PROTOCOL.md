# Experiment protocol - qwen4exp on RDNA4 (ROCm/HIP)

Single source of truth for how experiments are designed, run, and recorded on this
branch. Read this before touching `INDEX.md` or creating a run record.

Scope: performance and correctness work on the `qwen4exp` arch (HF
`Qwen/Qwen3.8-Flash-Next`) using the HIP backend on RDNA4 (gfx1201 / Navi 48).
Local branch only - nothing here is intended for submission as-is.

---

## 1. Directory layout

```
.pi/agent/memory/
  qwen4exp-rdna4.md            hub memory: state, machines, entry points
  <topic>.md                   durable facts (arch map, build recipe, backend quirks)
  experiments/
    PROTOCOL.md                this file
    INDEX.md                   every run, one row; the only place to scan for history
    plans/backlog.md           candidate hypotheses, prioritised, with status
    runs/E<nnn>-<slug>.md      one record per experiment
    results/E<nnn>-<slug>.*    raw, machine-generated output (never hand-edited)
    hw/<machine>.md            machine profiles (hardware, versions, pinned config)
```

Raw output stays out of the run record. The record holds the summary table and a
pointer to the raw file. Both are committed together.

## 2. Identity and naming

- ids are `E001`, `E002`, ... assigned in `INDEX.md` **before** the first run, so a
  planned experiment is referencable while still empty.
- file name: `runs/E001-<short-kebab-slug>.md`, slug stable across edits.
- one experiment = one hypothesis. A refinement of a hypothesis is a new id that
  lists `parent: E<nnn>`. Do not append unrelated runs to a record.
- a record may contain many *runs* (repeats) of the same measurement.

## 3. Hard rules

1. **State the hypothesis, the deciding metric, and the expected effect size before
   running.** Written into the record first, flags and all. A number that only
   supports a conclusion you already reached is not evidence.
2. **Baseline and variant are measured in the same session, interleaved A/B/A/B.**
   Never compare across sessions, machines, ROCm versions, or commits. Same-session
   interleaving is the only defence against clock/thermal drift on this hardware.
3. **Correctness gate precedes any perf claim.** A change that alters numerics must
   pass `test-backend-ops` for every touched op and a fixed-prompt logprob diff
   (`scripts/compare-logprobs.py`). Record the max/mean absolute delta. A faster
   wrong kernel is worth nothing, and "it still generated text" is not a check.
4. **Median of >= 3 repeats, with min-max spread reported.** If spread > 5% for a
   steady-state metric, the measurement is contaminated - find the cause and rerun.
   Never report a single run as a result.
5. **Negative results are recorded, not deleted.** Verdict `dead-end` entries are the
   most valuable thing in this directory: they stop the next person (or you) from
   re-testing the same idea.
6. **Raw data is committed with the record.** If it cannot be regenerated, it must
   exist. Logs, CSV/JSON from llama-bench, `--record` CSVs, kernel timings.
7. **Promote durable conclusions to memories.** The log is the audit trail; a memory
   is what survives context loss. When a run establishes a fact that will still be
   true next month, write it into a `<topic>.md` memory and link the run id.

## 4. Evidence tiers

Every claim in a record carries a tier. Optimization ideas that cannot produce a
tier-1 signal locally are suspect - usually noise, or an effect that only exists on
the model size you cannot run here.

| tier | where | what it can prove | cost |
|---|---|---|---|
| **T1** op-level | dev box (`hw/dev-rx9070-16g.md`) | correctness of a kernel; per-op throughput; fusion-pattern counts; graph-split counts; behaviour on a synthetic `qwen4exp` model | minutes |
| **T2** end-to-end | 4-GPU bench box (`hw/bench-4x-r9700-32g.md`), real weights | real pp/tg throughput, VRAM headroom, multi-GPU scaling, quality on long context | hours, manual |

T1 is authoritative for *does the mechanism work*; T2 is authoritative for *does it
matter*. A T2-only number (no T1 anchor) gets flagged `unmechanised` in the record -
it is a data point, not a finding, because the confounder set is not enumerable.

## 5. Measurement recipe

### 5.1 Fixed conditions (record every one, in the record's `conditions` block)

| knob | rule |
|---|---|
| commit | full sha; if the tree is dirty, `git diff` to `results/E<nnn>.patch` and say so |
| build | cache flags that differ from the profile default; never assume a rebuild happened |
| `LLAMA_NGL` / `-ngl` | fixed unless the experiment is about offload |
| `-c` / `-fd` | n_ctx fixed; context-fill state fixed (a warm vs cold KV cache changes pp) |
| `-b` / `-ub` | batch + ubatch fixed, they dominate pp |
| `-p` prompt | a fixed prompt file for tg tests, committed under `results/` |
| `-s` seed | fixed, per repeat |
| `-r` repeats | llama-bench `-r 5` for T2, `-r 3` minimum; always >= 3 |
| clocks | record `rocm-smi --showclocks` / `--showtemp` before and after; note if pinned |
| neighbours | no other GPU process, no compile jobs, no browser on the same GPU |

Any knob that changes between baseline and variant invalidates the comparison - say
so in the record even if the number still moves.

### 5.2 Standard metrics

- `pp<N>` t/s - prompt processing (compute-bound; the op-fusion and mma surface).
- `tg<N>` t/s - token generation (memory-bandwidth-bound; the quant/dequant and
  MoE-routing surface). These two respond to completely different bottlenecks; do not
  average them, do not trade one for the other silently.
- peak VRAM per device, from the profile log.
- graph splits / op counts from T1 harnesses (`test-fusion` CSV rows, graph dumps).
- wall-clock first-token latency for the server path, when the experiment is about
  interactive use.

Report effect as `median (+/- spread)` and `% vs baseline` on the same row.

### 5.3 T1 harness commands

Device naming on this backend is `ROCm0`, not `CUDA0` and not `HIP0` - verified with
`llama-cli --list-devices`. Every `--device` / `-b` filter below uses it.

Pin the loader first. The build is `BUILD_SHARED_LIBS=ON` with no rpath, and this box
has a stale llama.cpp install in `~/.local/lib64`, so an unpinned binary measures old
code (see `hw/dev-rx9070-16g.md`, "THE LOADER TRAP"):

```sh
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin
ldd build/bin/llama-bench | grep -E 'lib(ggml|llama)\.so\.0 '   # must resolve into build/bin
```

Then:

```sh
# fusion pattern counts per arch, per device, recorded as CSV
./build/bin/test-llama-archs -o /tmp/dummy-models             # synthetic models, all arches
./build/bin/test-fusion --models /tmp/dummy-models --device ROCm0 --record /tmp/f.csv
./build/bin/test-fusion --model  /tmp/dummy-models/qwen4exp-*.gguf --device ROCm0 --check /tmp/f.csv

# end-to-end on the synthetic qwen4exp model (fits in 16G)
./build/bin/llama-bench -m /tmp/dummy-models/qwen4exp-*.gguf -ngl 99 -r 3

# op support / correctness / timing (NOT currently usable - see below)
./build/bin/test-backend-ops support -b ROCm0 -o MUL_MAT
./build/bin/test-backend-ops perf    -b ROCm0 -o <OP> --output csv
```

`test-backend-ops` is reported to crash on AMD GPUs. Until that is triaged it is not
part of the standard gate, and rule 3 (correctness before perf) has to be satisfied
some other way - op-by-op on a synthetic model, or a logprob diff - and the record
must say which. Do not silently skip the correctness gate because the tool is broken.

The synthetic model is the whole point of the T1 tier: real `qwen4exp` weights do not
fit on the dev box, and downloading them is not possible on this SSD. A T1 number from
a synthetic model has the *same graph shape* and a different arithmetic payload - use
it for mechanism and relative deltas, never to predict absolute t/s.

## 6. Run record template

Copy to `runs/E<nnn>-<slug>.md`. Fill the top block *before* running.

```markdown
# E<nnn> - <title>

- date: YYYY-MM-DD
- machine: dev-rx9070-16g | bench-4x-r9700-32g
- tier: T1 | T2
- status: planned | running | done | blocked | dead-end
- parent: - | E<nnn>
- commit: <sha> (tree: clean | dirty -> results/E<nnn>.patch)
- build: <flags that differ from hw profile>
- model: <synthetic qwen4exp | real qwen4exp, quant, -ngl>

## hypothesis
<one sentence>

## prediction
<deciding metric, expected direction, expected effect size>
<what result would falsify it>

## conditions
<the 5.1 table, actual values, plus env vars>

## results
| variant | metric | median | min-max | vs base | n |
|---|---|---|---|---|---|

## raw
<paths under results/>

## verdict
<accepted | rejected | inconclusive + what is missing>

## notes
<mechanism, surprises, what this opens or closes>
```

## 7. Cross-machine workflow

The dev box cannot run the real model, and the bench box is not where code gets
written, so every T2 experiment is a hand-off. Keep the hand-off explicit:

1. record written on the dev box with `status: planned`, including the exact command
   block to run, copy-pasteable, on the bench box.
2. code/build state shipped as a commit sha + patch, never as "the current binary"
   (the bench box must be able to reproduce the sha).
3. run on the bench box, ship back the raw output verbatim into `results/`, no
   transcription of numbers into chat - transcribed numbers get typos that survive.
4. fill the results/verdict sections on the dev box, so the analysis happens in one
   place with the code in front of you.

## 8. Hygiene

- `.pi/` is tracked by upstream llama.cpp (`.pi/gg/SYSTEM.md`). These memory files are
  untracked-but-visible in `git status`. Never `git add -A` or `git add .` on this
  branch - always add explicit pathspecs, and never include `.pi/` in a commit that is
  destined for a PR.
- never commit model weights, gguf files, or anything > a few hundred KB. Synthetic
  models live in `/tmp`, and a record links the generating command instead.
- `export LD_LIBRARY_PATH=$PWD/build/bin` in every recorded command block, so the
  record is runnable by someone else without inheriting this box's stale `~/.local`.
- record the GPU idle state (free VRAM) at the start of every T1 run on the dev box -
  it is a display GPU, and a desktop eating 1.5 GB changes what fits in a batch.
- if the ROCm stack, kernel driver, or firmware on a box changes, bump `profile: vN`
  in that `hw/*.md` and add a row to `INDEX.md` under "comparability breaks". Numbers
  measured under two different profiles are never baselines for each other.
- keep this protocol shorter than useful: if a rule is not being followed, delete it
  rather than letting the file rot.
