# E062 - does the mmvf unroll really lose the wall, with an unprofiled alternating A/B

- date: -
- machine: bench-4x-r9700-32g
- tier: T2 (manual, short)
- status: planned
- parent: E061
- commit: as E061; the arms differ only by `#define MMVF_K_UNROLL 1|4`
- build: one TU rebuild per arm, as in E061
- model: real Qwen3.8-Flash-Next Q4_K_M, `-d 16384 -p 0 -n 128`

## hypothesis

E061's device accounting says the unroll removes 0.572 ms/step/card of mmvf work and the
allreduce's wait tail returns 0.433 ms/step/card, i.e. a net **-0.139 ms/step/card** of
device time. That predicts a small *gain*. E061 measured a 1.4-2% *loss* on the wall in the
same session, but all three of its arms ran with `GGML_CUDA_AR_ONESHOT_PROBE=1` (E059's A/B
did not set it) and every wall number there was profiled, which inflates the token period by
~45%. So one of two things is true and this decides which:

- the allreduce tail is an artifact of the probe plus the profiler, and the unroll is worth
  roughly what the device accounting says (a tie within the 5% bar, not a loss);
- the tail is real, and the unroll is a wall loss even without a profiler.

## prediction

Deciding metric: `tg128 @ d16384`, arms **alternating in one session**, `-r 10`, no profiler.
Report the per-rep spread as well, and the paired delta between arms.

| arm | build |
| --- | --- |
| A0 | `MMVF_K_UNROLL=1` |
| A1 | `MMVF_K_UNROLL=4` |

Falsified (i.e. the unroll is a real loss) if the paired delta is worse than -1%. It is
"no wall effect either way" if the delta is inside +/- 1%, which is the most likely outcome
and still useful: it means the kernel win is real but unexploitable while the barrier's tail
is unaddressed, which is exactly the conclusion H25 is about.

## conditions

- `GGML_CUDA_AR_ONESHOT_PROBE` **unset** (the point of the exercise), everything else as
  E059's T2 arms: `GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_BF16=nccl
  Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_SPARSE_FA=1`,
  `-lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU`.
- Alternating `for pass in 1 2; do for arm in 0 1; do ...; done; done`, `-r 10`.
- Record the `.so` md5 per arm; the two binaries must be E061's u1 and u4 arms.
- No profiler in this run, so no device-time reading here; that accounting is already in E061.

## results

To be filled when it runs.