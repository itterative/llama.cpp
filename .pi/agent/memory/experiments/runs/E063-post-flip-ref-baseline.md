# E063 - post-flip REF baseline: the anchor every later A/B is read against

- date: 2026-09-27
- machine: bench-4x-r9700-32g
- tier: T2
- status: done
- parent: E062 (the flip it anchors), bundle item 2
- build: run reported `build: b673ab4ae`, i.e. the tree *before* the default flip `8372ffc1f`, with the
  macro forced to 4 by hand. Code-identical to the flip (the flip only changes the default's value), but
  no md5 and no macro grep were captured, so the arm identity rests on the user's word.
- raw: [results/user/bench-finish-2026-09-27/](../results/user/bench-finish-2026-09-27/) (`commands.sh`,
  `ref-baseline-u4-r10.log`, the user's own test run, no profiler regions)

## the numbers

One invocation, `-r 10`, no tracer, 16 tests:

| test | d4096 | d16384 | d40960 | d131072 |
| --- | --- | --- | --- | --- |
| pp512 | 1478.04 +/- 181.48 | 1467.15 +/- 165.71 | 1405.50 +/- 140.96 | 1121.55 +/- 54.78 |
| pp4096 | 2143.56 +/- 2.61 | 2063.75 +/- 3.57 | 1913.38 +/- 2.93 | 1486.56 +/- 1.80 |
| pp8192 | 2219.27 +/- 3.16 | 2120.65 +/- 1.06 | 1958.24 +/- 1.15 | 1521.66 +/- 2.01 |
| **tg128** | **43.43 +/- 2.27** | **43.15 +/- 2.18** | **41.88 +/- 2.10** | **37.64 +/- 1.67** |

Depth curve, tg128: -0.6% at d16384, -3.6% at d40960, -13.3% at d131072 against d4096. The pp512 rows
carry a 5-12% spread while pp4096/pp8192 sit at 0.1-2%: the first test of each depth group pays the
depth-fill reserve, so pp512 is a warmup row and not a measurement.

Environment, verbatim in `commands.sh`: `GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal
GGML_CUDA_AR_DIRECT_BF16=nccl GGML_CUDA_AR_ONESHOT_PROBE=50 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1
Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1`, `-lm none -sm tensor -fa 1 -lzm on-direct
-ot per_layer_token_embd=CPU -d 4096,16384,40960,131072 -p 512,4096,8192 -n 128 -r 10 -b 2048 -ub 1024`.

## what it anchors, and what it does not

This is the level to read later A/Bs against **in the same session shape**: tg measured *after* the pp
tests at the same depth, probe flag set to 50, no tracer. Two things it does not license:

1. **It is not the bundle's REF environment.** The REF block in
   [plans/bench-finish-bundle.md](../plans/bench-finish-bundle.md) omits `GGML_CUDA_AR_ONESHOT_PROBE`
   deliberately (E061's arms set it to 1, and the untraced E062 logs do not echo the env at all), and
   this run has it at **50**. Read from the source, the value is an iteration count for a one-time
   startup self-test plus latency log (`allreduce-p2p.cu:403-500`, called once from pipeline setup at
   line 1019; the timings are logged and never feed the algorithm choice), so it should not matter
   steady-state - but it is the one variable that differs from the traced arms, and it is exactly the
   kind of thing that should not be left implicit in an anchor. E064 tests it.
2. **A tg-only invocation is not obviously the same measurement.** This run's tg came after pp at the
   same depth; the E062 arms measured tg with `-p 0`. The two agree on configuration and disagree by
   6.8% at d16384 (43.15 here against the U=4 arm's 40.41), which is ~9 standard errors apart, so it is
   not sampling noise. See "left open".

## cross-checks

**Prefill is stable across sessions; decode is not.** Against E050's session (2026-09-24, build
`06c3af843`, pool-on arm, `-r 3`):

| | d4096 | d16384 | d40960 | d131072 |
| --- | --- | --- | --- | --- |
| pp8192, this run vs E050 | 2219.27 vs 2246.95 (-1.2%) | 2120.65 vs 2138.89 (-0.9%) | 1958.24 vs 1959.43 (-0.1%) | 1521.66 vs 1508.89 (+0.8%) |
| tg128, this run vs E050 | 43.43 vs 32.44 (+33.9%) | 43.15 vs 31.98 (+35.0%) | 41.88 vs 31.64 (+32.4%) | 37.64 vs 30.12 (+25.0%) |

pp agreeing within 1.2% two days apart says the box itself is in the same state; the decode level moving
25-35% over the same span is the three decode-side default flips of that window (small-k `855a65544`,
pool `9111adf2c`, mmvf unroll `8372ffc1f`) plus E059's narrow-k change. That is the sharpest
confirmation yet that those flips delivered at the bench rather than only on the dev box, and it is also
why no pre-`9111adf2c` tg number may be quoted as comparable.

Against the same day's E062 arms: d16384 43.15 vs 40.41 (**+6.8%**). Against E059's session
(2026-09-26, build `d133df7d4`): d4096 43.43 vs 37.59 (+15.5%), d40960 41.88 vs 36.09 (+16.0%). The
E059 comparison is not matched either (different depth set, `-r 3`, one day apart), so it is context,
not evidence.

## left open

- **The 6.8% against E062's own U=4 arm.** Same configuration, same day, both `-r 10`, one of them
  tg-only and one tg-after-pp. Candidates, in order of how cheap they are to kill: (a) the test order
  and the state it leaves (depth fill reserve, pooled block keys, lazy-prefetch reuse) - a tg-only arm
  against a tg-after-pp arm in one session settles it; (b) the probe iteration count (E064); (c) box
  state. Until one of them is closed, read the anchor within a session and treat cross-session tg
  comparisons at fixed config as having a ~7% floor.
- The anchor needs re-capturing after any further default flip, and it should carry the bin/so md5 next
  time.