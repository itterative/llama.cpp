# E064 - is the allreduce startup probe count, or the test order, worth 7% of decode?

- date: -
- machine: bench-4x-r9700-32g
- tier: T2 (flag-only, no rebuild, 4 invocations)
- status: planned
- parent: E063 (the anchor whose two unexplained differences this tests)
- commit: any tree from `8372ffc1f` on (the flip is the default; nothing here is compile-time)

## hypothesis

E063's anchor measures 43.15 t/s at d16384 against E062's own U=4 arm at 40.41 - same configuration, same
day, both `-r 10`, ~9 standard errors apart. Two variables differ and either could own it:

- **H-probe.** E063 set `GGML_CUDA_AR_ONESHOT_PROBE=50`; E061's traced arms set 1 and the E062 untraced
  logs do not echo the env. Read from the source the value is only the iteration count of a one-time
  startup self-test whose timings are logged and never used (`allreduce-p2p.cu:403-500`, one call at
  pipeline setup), so it should be inert steady-state. But the loop it drives runs the one-shot
  *back-to-back over the real links*, which is exactly the kind of thing that leaves the peer inboxes,
  generations and link state warm, so it is not free by construction.
- **H-order.** E063's tg ran *after* three pp tests at the same depth, so the depth-fill reserve, the
  pooled block keys and the lazy-prefetch state were already paid; E062's arms measured tg with `-p 0`.
  E063's own pp512 rows carry a 5-12% spread against pp4096/pp8192's 0.1-2%, which shows the first test
  of a depth group pays something real.

## prediction

One session, four invocations, `-r 10`, no tracer, otherwise E063's environment exactly:

| arm | probe | tests |
| --- | --- | --- |
| A | unset | `-p 0 -n 128 -d 16384` |
| B | 50 | `-p 0 -n 128 -d 16384` |
| C | unset | `-p 4096,8192 -n 128 -d 16384` |
| D | 50 | `-p 4096,8192 -n 128 -d 16384` |

Deciding metrics: `tg128 @ d16384` for all four; `pp8192 @ d16384` as the control that the box did not
move between arms (E063 showed pp is stable to ~1% across days, so an arm-to-arm pp move means the
session drifted and the run is void).

- **H-probe falsified** if B differs from A, or D from C, by 3% or more - the probe is then a real lever
  and E061/E062's arms are not comparable to E063.
- **H-order confirmed** if C differs from A and D from B in the same direction, by 3% or more.
- Both inert (deltas inside +/-1.5%) means E063's anchor and E062's arms are comparable after all and the
  6.8% was session drift with no flag-level cause - which is itself worth recording, because it sets the
  cross-session floor for this box.

## conditions

```sh
BENCH="llama-bench -m $M -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -b 2048 -ub 1024"
REF="GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_BF16=nccl GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1"
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 0 -n 128 -d 16384 -r 10
GGML_CUDA_AR_ONESHOT_PROBE=50 $REF $BENCH -p 0 -n 128 -d 16384 -r 10
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 4096,8192 -n 128 -d 16384 -r 10
GGML_CUDA_AR_ONESHOT_PROBE=50 $REF $BENCH -p 4096,8192 -n 128 -d 16384 -r 10
```

Run them A, B, C, D in that order and then D, C, B, A again if time allows - order effects are one of the
two things under test, so a single pass cannot separate them from drift. Record the `.so` md5 (no rebuild
here, so all four arms must match) and the exact env of each invocation.

## results

To be filled when it runs.