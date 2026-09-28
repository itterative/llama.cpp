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

One session, `-r 10`, no tracer, otherwise E063's environment exactly. E063 itself also differed from
E062's arms in a third way, which the user raised: E062's untraced arms still had `GGML_PROF_REGIONS=1`
plus the roctx window (the logs say `regions on: counters, sink 'roctx`), and E063's own run had **no
profiling at all**. With ~300 region calls per token in the trace, that is not obviously free either, so it
gets an arm of its own. One factor at a time against a repeated control:

| arm | probe | tests | regions |
| --- | --- | --- | --- |
| A | unset | `-p 0 -n 128 -d 16384` | off |
| B | 50 | `-p 0 -n 128 -d 16384` | off |
| C | unset | `-p 4096,8192 -n 128 -d 16384` | off |
| D | unset | `-p 0 -n 128 -d 16384` | `GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1` |
| E | 50 | `-p 4096,8192 -n 128 -d 16384` | `GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1` |
| A' | unset | `-p 0 -n 128 -d 16384` | off |

Deciding metric: `tg128 @ d16384` in all six. `pp8192 @ d16384` is the control that the box did not move
between arms (E063 showed pp is stable to ~1.2% across days, so an arm-to-arm pp move voids the session).
A against A' bounds the drift; anything inside that band is not a finding.

- **H-probe falsified** if B differs from the A/A' band by 3% or more - the probe is then a real lever and
  E061/E062's arms are not comparable to E063.
- **H-order**: C against the A/A' band.
- **H-regions**: D against the A/A' band. E is E063's exact shape (regions on, probe 50, tg after pp), so
  it should reproduce ~43.15 if the three main effects add up - which is the check that this design
  explains the 6.8% rather than just measuring parts of it.
- All three inert means E063's anchor and E062's arms are comparable after all and the 6.8% is session
  drift with no flag-level cause - itself worth recording, because it sets this box's cross-session floor.

## conditions

```sh
BENCH="llama-bench -m $M -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -b 2048 -ub 1024"
REF="GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_BF16=nccl GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1"
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 0 -n 128 -d 16384 -r 10                     # A
GGML_CUDA_AR_ONESHOT_PROBE=50 $REF $BENCH -p 0 -n 128 -d 16384 -r 10                        # B
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 4096,8192 -n 128 -d 16384 -r 10            # C
env -u GGML_CUDA_AR_ONESHOT_PROBE GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1 $REF \
    $BENCH -p 0 -n 128 -d 16384 -r 10                                                       # D
GGML_CUDA_AR_ONESHOT_PROBE=50 GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1 $REF \
    $BENCH -p 4096,8192 -n 128 -d 16384 -r 10                                               # E
env -u GGML_CUDA_AR_ONESHOT_PROBE $REF $BENCH -p 0 -n 128 -d 16384 -r 10                    # A'
```

Run them in the listed order and then A' last, or interleave A between each other arm if the whole set is
run twice - order is one of the things under test, so a single un-repeated pass cannot separate it from
drift. Record the `.so` md5 (no rebuild here, so every arm must match) and the exact env of each
invocation. D and E print their own region report; keep it, the region-call counts are the mechanism check
for H-regions.

## results

To be filled when it runs.