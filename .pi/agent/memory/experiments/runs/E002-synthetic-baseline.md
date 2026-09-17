# E002 - reproducible synthetic qwen4exp baseline on ROCm0

- date: -
- machine: dev-rx9070-16g
- tier: T1
- status: dead-end
- parent: E001
- commit: -
- build: as E001, plus whatever B0 decides (static vs LD_LIBRARY_PATH pin)
- model: synthetic qwen4exp from `test-llama-archs`, `-ngl 99`

## hypothesis

A synthetic `qwen4exp` model loads on `ROCm0` and yields a pp/tg baseline stable enough
(spread <= 5%) to serve as the reference for all later T1 deltas.

## prediction

`llama-bench -r 5` repeats over the synthetic model give `pp` and `tg` medians with
min-max spread <= 5% across two separate sessions. Falsified if spread exceeds 5%, or
if the run's op profile shows any op falling off the GPU (then the baseline measures the
CPU fallback, not the thing we want to tune).

## conditions

To be filled from `PROTOCOL.md` 5.1 at run time. Fixed and recorded explicitly: n_ctx,
n_batch, n_ubatch, prompt file, seed, `-ngl 99`, repeats, GPU idle state, clocks.

## planned command block

```sh
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin
cmake --build build -j"$(nproc)"
ldd build/bin/llama-bench | grep -E 'lib(ggml|llama)\.so\.0 '    # must show build/bin

rm -rf /tmp/dummy-models && mkdir -p /tmp/dummy-models
./build/bin/test-llama-archs -o /tmp/dummy-models
ls -la /tmp/dummy-models | grep -i qwen4exp

# does the graph stay on the GPU, and what shape is it?
./build/bin/test-fusion --models /tmp/dummy-models --device ROCm0 --record /tmp/fusion-v1.csv

# the baseline itself, two sessions
for i in 1 2; do
  ./build/bin/llama-bench -m /tmp/dummy-models/qwen4exp-*.gguf -ngl 99 -r 5 -o csv \
    >> results/E002-llama-bench.csv
done
```

## results

| variant | metric | value | note |
|---|---|---|---|
| synthetic qwen4exp (4.80 M params, 19.24 MB, F32, 2 layers, ngl 99) | pp32 | 11768 t/s | **not a baseline** |
| same | tg8 | 325 t/s | not a baseline |
| same, `-fa 1` | pp32 / tg8 | 11490 / 318 t/s | FA on is within ~2% here, i.e. this regime does not exercise FA at all |

Single repeats only (`-r 1`); no spread was computed because spreading was never the question.

## verdict

**rejected - the hypothesis was wrong, and it is worth being explicit about why.** A
synthetic qwen4exp model cannot yield a usable pp/tg baseline, because its entire weight set
(19 MB, F32) fits in on-die cache. `pp` measures how fast the harness launches kernels and
`tg` measures per-token fixed cost - input-path construction, graph overhead, host-side
work. Neither is the real model's bottleneck, which is streaming ~80 GB of Q4_K_M weights
per token (`tg`) and iterating KV at 262 k context (`pp`).

What was actually true in the hypothesis is that the synthetic model is *stable and
reproducible* - it is. It just measures the wrong thing. Confirmed by the two runs above
agreeing to ~2% and by `test-llama-archs` being deterministic.

So the synthetic model keeps exactly the roles E001 established - graph executes on
`ROCm0`, op coverage, no-tokenizer caveat - and loses the baseline role entirely.

Also pending: the qwen4exp row count in `/tmp/fusion-v1.csv` (which fusions fire on
ROCm0 vs which do not), and the model file size.

## raw

`results/E002-llama-bench.csv`, `/tmp/fusion-v1.csv` (copy into `results/` at record time).

## verdict

pending

## notes

Two things this record must capture, because every later experiment inherits them:
the synthetic model's dimensions (so a later "why is this so much smaller than the real
model" question has an answer), and the list of ops that do not run on GPU. Both are now in
E001 update 2 and `rdna4-rocm-build`.

What replaces this experiment:

- **E004, if we want a T1 perf instrument at all**: a config-shaped synthetic model - real
  `head_dim` 256, `24/2` heads, `hc_lowrank` 320, `indexer_budget` 2048, more layers, fewer
  experts to fit 16 GB. Requires editing dims in `tests/test-llama-archs.cpp` (a local
  source change, not a new test file). Then `pp` context-scaling becomes measurable and
  E003's QSA tax can actually be quantified. Until then E003 has no instrument.
- **otherwise perf is T2-only**, on the bench box with real weights, with whatever
  handoff cost that implies.

Rejected alternative considered mid-run: chasing graph splits from the 325 t/s figure.
Dropped as unproductive - see E001 update 2.
