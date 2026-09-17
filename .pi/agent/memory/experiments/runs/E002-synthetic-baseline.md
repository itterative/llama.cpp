# E002 - reproducible synthetic qwen4exp baseline on ROCm0

- date: -
- machine: dev-rx9070-16g
- tier: T1
- status: planned
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

| variant | metric | median | min-max | vs base | n |
|---|---|---|---|---|---|
| (pending) | | | | | |

Also pending: the qwen4exp row count in `/tmp/fusion-v1.csv` (which fusions fire on
ROCm0 vs which do not), and the model file size.

## raw

`results/E002-llama-bench.csv`, `/tmp/fusion-v1.csv` (copy into `results/` at record time).

## verdict

pending

## notes

Two things this record must capture, because every later experiment inherits them:
the synthetic model's dimensions (so a later "why is this so much smaller than the real
model" question has an answer), and the list of ops that do not run on GPU. If the op
list is non-empty, that list *is* the first optimisation target and B3/B4/B7 may be
moot.
