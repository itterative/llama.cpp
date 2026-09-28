# E075 - llama-bench does show the realloc, and the fix is +16.4% pp on the dev box

Status: **closed** for the measurement, **open** for the attribution. Dev box (`dev-rx9070-16g`, 1 GPU), the
4-layer test model `models/q4exp-4l.gguf`, `GGML_PROF_REGIONS=1`, same command on both commits:

    llama-bench -m models/q4exp-4l.gguf -fa 1 -ngl 99 -p 8192 -n 0 -r 3 -b 2048 -ub 1024

Raw logs: scratchpad `h19-lb-before.log` (pre-fix), `h19-lb-head.log` (fix), `h19-lb-<arm>.log` (ablation).

## 1. Where llama-bench shows the realloc

Two places, and they are the same cost seen twice:

- `[prof] sched:realloc_size` - **rows only exist when the region was hit**, so an absent row is a true zero.
  This is what makes the bench-box logs readable: the same pre-fix commit prints the row on the dev box, so its
  absence there is a real zero rather than a blind instrument.
- `graph:alloc` - the reallocation is inside it, so a reallocation-heavy run makes `graph:alloc` per-call cost
  jump by two orders of magnitude (below).

## 2. Before and after the fix

| row | before `6866b5fc1` | after `2c016ed3d` (HEAD) |
| --- | --- | --- |
| pp8192 t/s | 13248.27 ± 477.36 | **15419.44 ± 129.00** (+16.4%) |
| `graph:alloc` (measured) | 24 calls, 1220.82 ms (**50.87 ms/call**) | 24 calls, 3.88 ms (0.16 ms/call) |
| `sched:realloc_size` | **24 calls, 1218.55 ms** (50.77 ms/call) | row absent |
| `graph:compute` | 24 calls, 349.55 ms | 24 calls, 1298.05 ms |
| `phase:prefill` | 12 calls, 1670.42 ms | 12 calls, 1403.13 ms |
| regions registered | 8 | 7 |

**Every single allocation re-reserved** pre-fix (24/24 in the measured section, and 8/8 in warmup), and the
reallocations were ~73% of the measured prefill wall (1218.55 of 1670.42 ms). This is the cleanest measurement
of the H19 fix so far: +16.4% pp on a run where it applies, against 0% for the bench box's dense prefill (E074).

## 3. The flag ablation, to attribute the bench box's silence

Pre-fix build, box flags added one at a time. **Every arm still reallocates**, so none of these is why the
bench box sees nothing:

| arm | reallocs | ms/call | t/s |
| --- | --- | --- | --- |
| baseline (no box flags) | 24 | 50.77 | 13248.27 ± 477.36 |
| `-lzm on-direct` | 24 | 49.81 | 11363.25 ± 126.13 |
| `-lm none` | 24 | 50.71 | 13268.60 ± 437.62 |
| `-d 16384` | 40 | 54.95 | 11431.24 ± 23.43 |
| `-ot per_layer_token_embd=CPU` | 24 | 50.90 | 13175.55 ± 491.13 |
| all four together | 40 | 54.70 | 9623.81 ± 204.81 |

`-lm` is `--load-mode <auto|none|mmap|mlock|mmap+mlock|dio>`, not a logits flag (the log column is confusingly
labelled `lm`); `-lzm` is `--lazy-mode <on|on-direct|auto|off>`. `on-direct` does switch the lazy path on - the
arm registers 13 regions and its log matches the bench box's region list - and it still reallocates, so the
loading path does not mask the ratchet either.

## 4. What is left to attribute it

After the ablation the bench box still differs from every dev-box arm in exactly two ways:

1. **The model**: 19.73 B 4-layer test model against the real 111 GiB Qwen3.8-Flash-Next. The dep that starts
   all of this is reported per matching MoE weighted reduction, so a different last layer can plausibly remove
   the mismatch.
2. **The 4-GPU `-sm tensor` split.** The dep node the scheduler builds is a keep-alive view for a dependency
   that a split graph may already represent, so a split graph may not change its node count at all.

Both are cheap to separate on the bench box, on the instrumented pre-fix build `32ba4c666` (which has the
per-ubatch node count and the dep dumps; `6866b5fc1` predates them):

- one `pp4096 @ d16384` run with `GGML_ALLOC_DEBUG_REALLOC=1 LLAMA_UBATCH_DEBUG=1 -v`: compare the printed
  reserve count (6825) against the runtime per-ubatch count, and read the dep maps. That says directly whether
  the fusion reports its dep in both graphs there, which is the whole question;
- or the same test under `-sm layer` against `-sm tensor`: if the split mode flips it, it is about splits, not
  about the number of GPUs.

Until one of those is run, "single GPU" is a plausible hypothesis, not a finding: the flag axis is measured and
excluded, the model and split axes are not.