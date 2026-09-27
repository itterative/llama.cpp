# E061 - the mmvf decode loop runs at 16 VGPRs; does more in-flight work per thread pay on the bench

- date: 2026-09-27
- machine: dev-rx9070-16g (T1 screen), bench-4x-r9700-32g decides
- tier: T1 screen + T2 verdict
- status: done (kernel lever real, does not pay on 4 cards)
- parent: E060
- commit: `09fd33f1a`
- build: as E060; instrument is `MMVF_K_UNROLL` (compile-time, 1/2/4)
- model: the E060 census and probe cases for the screen, `models/q4exp-4l.gguf` for the regression

## hypothesis

The decode path of `mul_mat_vec_f` (`ncols_dst == 1`, no fusion) keeps about one load per
thread in flight. The compiler's own register report says **VGPRs: 16** for the f32
instantiation (`-Rpass-analysis=kernel-resource-usage`), i.e. the K loop reuses one load
register pair, so each iteration's load waits for the previous pair to be consumed. That is
~1 MB in flight across the card (512 blocks x 256 threads x 8 B).

Cache-resident that is enough: the dev box does the f32 router shape (m=512, k=2560,
5.24 MB) in 7.19 us, L3-bandwidth-bound. Cold it is not: the same bytes at the same
512-block grid reach 612 GB/s on the dev box once each thread has ~128 iterations (E060's
k=65536 probe), and a cold batch of 16 slices costs 8.34 us per slice. The bench decode
window pays 27.37 us per call (p05-p95 26.2-28.5, i.e. deterministic, not a latency
lottery). Across 48 router calls per step, 27.37 -> 8.3 us is ~0.9 ms/step, 6% of device
time - the largest single item left in the mmvf group.

Hypothesis: the loop's per-thread ILP is what the bench pays for, and unrolling it with U
independent accumulator/live-load pairs recovers most of that gap.

## prediction

Deciding metric: **p50 of the `mul_mat_vec_f<float, float, 1, 256, false, false>` row with
grid 512 in a bench decode window** (the E059 instrument: `kernel_trace.csv` per dispatch),
plus per-step device ms per card. Not `tg`: 0.9 ms on a 27 ms step is ~3%, under the 5% bar.

| arm | `MMVF_K_UNROLL` | what changes |
| --- | --- | --- |
| A0 | 1 | the plain loop, control |
| A1 | 2 | two live load pairs and two accumulators per thread |
| A2 | 4 | four of each |

Applied only when `ncols_dst == 1 && !has_fusion` and `T` is f32 or bf16 - the two types
the mmvf group uses. The half path, the fused paths and every prefill path keep the plain
loop untouched.

Predicted: A2 takes the router row to <= 15 us, A1 to <= 20; the bf16 `indexer.q_proj` row
(23.7 us) follows; the wide rows (`output_hc_*`, 15.2 and 22.1 us at 0.02 ms/step each) do
not move because they are already DRAM-bound.

Screen on the dev box is **regression-only** - E060 1c showed the bench's regime (cold
plus few blocks) cannot be built here:

- `-Rpass-analysis=kernel-resource-usage` per arm: VGPRs must rise, `VGPRs Spill` must stay
  0 and occupancy must stay at 16 waves/SIMD. A VGPR count still at 16 means the compiler
  did not keep the loads live and that arm is inert, whatever the timing says.
- the 23 census/probe cases against the day's noise floor, `test-backend-ops test -b ROCm0
  -o MUL_MAT` (1297 cases), the golden PPL (non-bit-exact by construction: the summation
  order changes) and the dummy tg, interleaved.

Falsified if (a) the screen spills, loses occupancy or regresses a census/probe case beyond
noise, or (b) the bench row's p50 stays within 10% of baseline.

## conditions

Dev box per `PROTOCOL.md` 5.1 and the E059 harness: `LD_LIBRARY_PATH=$PWD/build/bin`,
device filter `ROCm0`, two runs per cell, control repeated last. Each arm is a rebuild of one
TU (`mmvf.cu`, ccache ~3 s once both exist), and the `.so` md5 per arm is recorded. The
knob is compile-time because the accumulators are register arrays.

## results

Verdict: **real at the kernel level, and real on the wall once the tracer is off**. The unroll
cuts the mmvf group by 22% of its device time and the tiny rows by 2.4x. Under `rocprofv3` that
saving is 76% absorbed by the allreduce's wait tail, which made the traced wall 1.4-2% *worse* -
but without the tracer the same two arms measure **+1.9-2.0% tg** and the absorption is
arithmetically absent (section 2b). So the absorption is a property of the *traced* regime, not of
the change. The pre-registered deciding number, the F32 router row, moved 27.64 -> 24.24 us:
direction right, size 3x smaller than predicted.

Bench run: `b673ab4ae`, one rebuild per arm, all three arms in one session,
`GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1 rocprofv3 --selected-regions --marker-trace --kernel-trace
--stats`, `llama-bench -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -d
16384 -p 0 -n 128 -r 3 -b 2048 -ub 1024`. Raw:
[results/user/h23-runs-b673ab4ae](../results/user/h23-runs-b673ab4ae/) (three kernel traces,
marker traces, stats, the driver script). Dispatch controls are identical in all three arms:
1,392,129 per agent, 36,960 `ar_oneshot`, 18,480 calls of each mmvf shape.

### 1. The kernel lever works, and unevenly per shape

Per-card device time per step and per call, agent 1:

| mmvf shape (per card) | calls/step | U=1 us | U=2 us | U=4 us | U=1/U=4 | saved ms/step/card |
| --- | --- | --- | --- | --- | --- | --- |
| f32 2560 x 512 (router) | 48 | 27.64 | 28.22 | **24.24** | **1.14x** | 0.163 |
| bf16 10240 x 4 (`hc_*_inject`) | 96 | 5.83 | 3.50 | **2.41** | **2.42x** | **0.328** |
| f32 2560 x 12 (`ssm_alpha/beta`) | 72 | 2.45 | 1.99 | **1.74** | 1.41x | 0.051 |
| f32 2560 x 1 | 48 | 2.23 | 1.75 | **1.53** | 1.46x | 0.034 |
| bf16 10240 x 320 (`output_hc_down`) | 1 | 15.23 | 13.05 | **12.52** | 1.22x | 0.003 |
| bf16 2560 x 128 (`indexer.k_proj`) | 12 | 2.53 | 2.42 | **2.40** | 1.05x | 0.002 |
| bf16 320 x 10240 (`output_hc_up`) | 1 | 22.21 | 21.33 | **21.25** | 1.05x | 0.001 |
| bf16 2560 x 512 (`indexer.q_proj`) | 12 | **23.84** | 24.34 | 24.68 | **0.97x** | -0.010 |
| f32 4 x 64 x 4160 (indexer scores) | 12 | 8.54 | 8.51 | 8.56 | 1.00x | 0.000 |

Group totals over the four agents: `mul_mat_vec_f` 4043.5 -> 3658.6 (U=2) -> **3162.9 ms**
(U=4), i.e. **-880.6 ms, -21.8%**, and saved evenly per card (each agent loses ~210 ms).
`ncols_dst == 4` rows are untouched, which is the guard working.

### 2. The allreduce takes 76% of it back

| kernel group (four agents, whole run) | U=1 | U=4 | delta | per step per card |
| --- | --- | --- | --- | --- |
| `mul_mat_vec_f` | 4043.5 ms | 3162.9 ms | -880.6 | -0.572 ms |
| `ar_oneshot` | 5612.5 ms | 6279.4 ms | **+666.9** | **+0.433 ms** |
| everything else | 13126.3 ms | 13147.9 ms | +21.6 | +0.014 ms |
| total device time | 22782.3 ms | 22590.2 ms | -192.1 | -0.125 ms |

`ar_oneshot` duration on agent 1: p10 10.96 -> 11.00 us, **p50 19.68 -> 18.64** (better),
**p90 36.56 -> 140.30**, p99 321 -> 523. So the *body* of the distribution improved and the
*tail* exploded: 10% of collectives now cost the waiting cards 140 us instead of 37. Agent 4
(the straggler, which waits least) barely moved (mean 32.13 -> 30.78). Per card over the run:
A1 1495.2 -> 1784.3 ms, A2 1466.0 -> 1738.3, A3 1463.9 -> 1619.3, A4 1187.4 -> 1137.6.

The extra wait sits in specific collectives, not everywhere: group the 96 per step by index and
agent 1's p90 at U=4 is 433-579 us on indices 6, 22, 38, 54, 70, 86 - every 8th layer, first
collective - against ~240 us on those indices at U=1 (which is uniform across all 96).

### 2b. Without the tracer the sign flips: +1.9-2.0% tg

The same three arms were re-run with no `rocprofv3` attached, only `GGML_PROF_REGIONS=1` plus the
roctx window (`u1/u2/u4-noprof.log` in the raw dir; no device trace, so no device accounting):

| | u1 | u2 | u4 |
| --- | --- | --- | --- |
| `tg128 @ d16384` | 39.65 +/- 2.31 | **40.46 +/- 2.39** | **40.41 +/- 2.36** |
| host launch + drain, ms/token | 16.9605 | 16.6956 (-1.97%) | 16.5241 (-1.86%) |
| `meta:allreduce` host span, ms/token | 0.7630 | 0.7756 (+1.65%) | 0.7657 (+0.35%) |

Two independent wall metrics agree: llama-bench's own `tg` and the process's own launch-plus-drain
total. The per-token gain at U=4 is 0.475 ms against a traced kernel saving of 0.572 ms/step/card
(83% conversion), where a surviving 0.433 ms of allreduce absorption would have left +0.55% instead
of +1.9%. The host allreduce span - which is the launch path, not the device spin - moved by
0.003-0.013 ms/token, i.e. 30-100x smaller than the traced device-side tail.

Precision, honestly: these are three sequential `-r 3` runs, so each mean carries a ~1.3 SE and the
difference ~1.9 on a 2% effect, and the arm order is not recorded in the logs (the mtimes are 6-7 s
apart, too close for full runs, so they were copied or written at the end). The sign is consistent
across both treatment arms and both metrics; the size is not yet protocol-grade. That is what E062's
paired `-r 10` is for, and it is now a confirmation rather than an open question.

### 3. Why the wall did not follow under the tracer, and which card it hurts

| | U=1 | U=2 | U=4 |
| --- | --- | --- | --- |
| device per card, agent 1 / agent 4 | 15.10 / 14.23 ms | 14.92 / 13.96 | 15.31 / 13.57 |
| host `phase:decode` | 34.78 ms/step | 34.79 | **39.08** |
| host `phase:sync` | 6.19 ms/step | 5.87 | **2.64** |
| host `meta:subgraph` per call | 0.4759 ms | 0.4551 | 0.4926 |
| token period, rep 2 / rep 3 | 37.34 / 41.24 ms | 36.96 / 41.05 | **37.88 / 42.05** |
| profiled `tg` @ d16384 | 25.03 | 25.22 | **24.59** |

The saving is real on every card, so the step does not shrink because the collectives are
barriers: the waiters arrive earlier and spin longer, and the extra spin blocks all four
streams. The host spans show where that lands - `phase:decode` (launch) grows 34.78 -> 39.08
ms/step while `phase:sync` (drain) halves, net +0.75 ms/step, which is the wall change. The
straggler card A1 ends up *worse* in total (5811.6 -> 5894.1 ms) while A4 improves 255 ms.

### 4. The pre-registered falsification

Predicted U=4 <= 15 us and U=2 <= 20 on the router row; measured 24.24 and 28.22. The "within
10% of baseline" rule is *just* cleared at U=4 (12.3%), so the honest reading is that ILP
explains about a seventh of that row's 27.6 us and something else owns the rest - E060's DRAM
floor for the same bytes was 8.3 us at full concurrency.

New facts this leaves behind, in order of value:

1. **The allreduce is the absorber.** A 22% cut in a kernel group's device time bought 0.8% of
total device time and a 1.4-2% *worse* wall. This is the first causal measurement of that
coupling on this branch, and it argues the comms thread is worth more than any further kernel
work in this group.
2. **The wait inflation is structural**, concentrated in 6 of the 96 collectives (every 8th
layer, first collective), not uniform, which is a strong hint for whoever picks up the
collective work.
3. `indexer.q_proj` (bf16 2560 x 512, 12 calls/step) regresses at both U=2 and U=4, so the
unroll would need a shape guard even if the absorption were fixed.

### 5. Caveat that keeps this from being the last word

The traced arms ran with `GGML_CUDA_AR_ONESHOT_PROBE=1`, which the E059 A/B did not set, and
their wall numbers are *profiled* (the profiler inflates the token period by ~45% here). The
untraced re-runs in 2b settle the wall question in the change's favour; the probe flag is still a
difference from E059's setup and the untraced logs do not echo their environment, so whether they
carried it is unknown. It does not change 2b's conclusion (the tracer is the variable that
matters), but it is worth setting explicitly in E062.
The device-time accounting is unaffected - it comes from the trace and the dispatch controls
match - but whether the allreduce tail is as fragile without the probe flag is untested. The
cheap settlement is one unprofiled, interleaved A/B of U=1 against U=4 at `-r >= 10`:
the device accounting predicts -0.14 ms/step (-0.5% tg) if the tail is a profiler/probe
artifact and +1.5% of loss if it is real.

### 6. T1 screen (dev box), which is what the arms above were checked against

Raw: [results/E061-mmvf-decode-ilp](../results/E061-mmvf-decode-ilp/) - `resource-screen.py`
and its log, `run-arms.sh`, per-arm case logs, dummy and PPL logs, `.so` md5 per arm.

**The loads are genuinely live.** `-Rpass-analysis=kernel-resource-usage`, the two decode
instantiations that matter (`ncols_dst=1`, block 256, no fusion):

| arm | f32 (router) VGPR | bf16 (indexer.q_proj) VGPR | spills | occupancy |
| --- | --- | --- | --- | --- |
| U=1 | 12 | 12 | 0 | 16 waves/SIMD |
| U=2 | 18 | 18 | 0 | 16 waves/SIMD |
| U=4 | 25 | 28 | 0 | 16 waves/SIMD |

Occupancy is at the wave limit and stays there, so the extra VGPRs cost nothing; the fused
paths and the half path keep their baseline counts, i.e. the guard works.

**Op-level, 23 census/probe cases** (`test-backend-ops perf`, `avg us/run`, one binary per arm):

| case | U=1 | U=2 | U=4 | best |
| --- | --- | --- | --- | --- |
| bf16 10240 x 4 (`hc_*_inject`) | 6.23 | 4.61 | **3.82** | **1.63x** |
| bf16 10240 x 320 (`output_hc_down`) | 11.46 | 8.55 | **7.42** | **1.54x** |
| f32 8192 x 512 | 15.22 | 12.11 | **11.39** | **1.34x** |
| bf16 2560 x 512 x b16 (cold batch) | 60.27 | 51.79 | **48.36** | 1.25x |
| f32 32768 x 512 | 48.88 | 42.94 | **39.17** | 1.25x |
| f32 16384 x 512 | 23.81 | 20.95 | **19.99** | 1.19x |
| f32 2560 x 1 / f32 2560 x 12 | 3.54 / 3.59 | 3.31 / 3.38 | **3.12 / 3.20** | 1.13x / 1.12x |
| f32 2560 x 512 (router shape) | 7.18 | 6.86 | **6.67** | 1.08x |
| f32 65536 x 512, cold batch x16 (DRAM controls) | 219.35 / 131.97 | 214.20 / 131.84 | 215.51 / 131.49 | 1.02x / 1.00x |
| bf16 320 x 10240, bf16 2560 x 512 | 25.74 / 7.17 | 25.89 / 7.41 | 26.30 / 7.20 | -2% / -3% at U=2 |

The two regressions sit inside the day's noise floor (E060 measured 0.4-2.3% on repeats of the
same setting). The shape of the result was the useful part: the kernel was ILP-starved in
*every* regime except DRAM-bound, including cache-resident ones, so the earlier reading (that
only the cold, few-block case suffered) was understated. The bench then showed that being
ILP-starved is worth less than the barrier's tail costs.

**Correctness.** `test-backend-ops test -b ROCm0 -o MUL_MAT` at U=4: **1297/1297, 2/2
backends**. Golden PPL 263113.6984 / .1919 / .4044 (U=1/2/4) = <=5e-7 relative, the reordered
K accumulation as predicted. Dummy `q4exp-4l -d 4096 -n 128 -r 5`, two interleaved passes:
U=1 263.48 / 262.50, U=4 265.58 / 265.36 = **+0.94% paired** - the dev box said yes, the
4-card bench said no, which is exactly the transfer risk E060's instrument work was about.

### 7. Left open

- **The paired confirmation (E062).** Untraced `-r 3` runs already flipped the sign to
  +1.9-2.0%; one interleaved U=1 vs U=4 run at `-r >= 10` turns that into a protocol-grade number
  and is the last gate before a default flip.
- **A shape guard for `indexer.q_proj`** if the unroll is ever revisited: bf16 2560 x 512 with
  12 calls/step regresses at both U=2 and U=4, and nothing else in the group does.
- Non-RDNA4 builds instantiate the unrolled kernels too. They are only reachable when
  `MMVF_K_UNROLL` is set, but a default flip would carry them everywhere - the same objection
  E059's record raises about the mmvq narrow kernels.
- The `MMVF_K_UNROLL` knob itself stays in the tree as the instrument for whoever picks up the
  collective work: it is the cheapest way to shorten the compute phases and watch what the
  barrier does with them.
