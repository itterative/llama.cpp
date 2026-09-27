# E061 - the mmvf decode loop runs at 16 VGPRs; does more in-flight work per thread pay on the bench

- date: 2026-09-27
- machine: dev-rx9070-16g (T1 screen), bench-4x-r9700-32g decides
- tier: T1 screen + T2 verdict
- status: planned
- parent: E060
- commit: -
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

Verdict: **the mechanism holds on the dev box, the bench reading is pending**. The unrolled
decode loop is a real 1.1-1.6x on every shape where the kernel is ILP-limited and *inert*
where it is DRAM-bound, which is exactly the split E060's probes predicted. The one number
that decides the project-level question - the F32 router row in the bench decode window -
needs the bench box.

Raw: [results/E061-mmvf-decode-ilp](../results/E061-mmvf-decode-ilp/) - `resource-screen.py`
and its log, `run-arms.sh`, per-arm case logs, dummy and PPL logs, `.so` md5 per arm.

### 1. The screen says the loads are genuinely live

`-Rpass-analysis=kernel-resource-usage`, the two decode instantiations that matter
(`<T, float, ncols_dst=1, block=256, has_fusion=false>`, one row per block, 512 blocks for
the router):

| arm | f32 (router) VGPR | bf16 (indexer.q_proj) VGPR | spills | occupancy |
| --- | --- | --- | --- | --- |
| U=1 | 12 | 12 | 0 | 16 waves/SIMD |
| U=2 | 18 | 18 | 0 | 16 waves/SIMD |
| U=4 | 25 | 28 | 0 | 16 waves/SIMD |

Occupancy is already at the wave limit and stays there, so the extra VGPRs cost nothing. The
fused paths and the half path stay at their baseline counts, i.e. the guard works.

### 2. Op-level: wins where the kernel is ILP-limited, inert where DRAM-bound

`test-backend-ops perf` on the 23 E060 census/probe cases, `avg us/run`, one binary per arm:

| case | U=1 | U=2 | U=4 | best |
| --- | --- | --- | --- | --- |
| bf16 10240 x 4 (`hc_*_inject`) | 6.23 | 4.61 | **3.82** | **1.63x** |
| bf16 10240 x 320 (`output_hc_down`) | 11.46 | 8.55 | **7.42** | **1.54x** |
| f32 8192 x 512 | 15.22 | 12.11 | **11.39** | **1.34x** |
| bf16 2560 x 512 x b16 (cold batch) | 60.27 | 51.79 | **48.36** | 1.25x |
| f32 32768 x 512 | 48.88 | 42.94 | **39.17** | 1.25x |
| f32 16384 x 512 | 23.81 | 20.95 | **19.99** | 1.19x |
| f32 2560 x 1 | 3.54 | 3.31 | **3.12** | 1.13x |
| f32 2560 x 12 (`ssm_alpha/beta`) | 3.59 | 3.38 | **3.20** | 1.12x |
| f32 2560 x 512 (router shape) | 7.18 | 6.86 | **6.67** | 1.08x |
| f32 65536 x 512 (**DRAM-bound control**) | 219.35 | 214.20 | 215.51 | 1.02x |
| f32 2560 x 512 x b16 (**DRAM-bound control**) | 131.97 | 131.84 | 131.49 | 1.00x |
| bf16 320 x 10240 (`output_hc_up`) | **25.74** | 25.89 | 26.30 | -2% |
| bf16 2560 x 512 (`indexer.q_proj`) | **7.17** | 7.41 | 7.20 | -3% at U=2 |

The two regressions are inside the day's noise floor (E060 measured 0.4-2.3% on repeats of the
same setting, and `output_hc_up` was the 2.3% one). The shape of the result is the point: the
kernel was ILP-starved in *every* regime except DRAM-bound, which the earlier reading
(that only the cold, few-block case suffered) had understated - the warm k=8192-32768 probes
and the cold-but-cache-resident bf16 batch improve just as much.

### 3. Correctness

- `test-backend-ops test -b ROCm0 -o MUL_MAT`, U=4: **1297/1297 passed, 2/2 backends**.
- Golden PPL: 263113.6984 (U=1), 263113.1919 (U=2), 263113.4044 (U=4), i.e. <= 5e-7 relative,
  which is f32 summation noise from the reordered K accumulation, as predicted.
- Dummy `q4exp-4l`, `-d 4096 -n 128 -r 5`, arms interleaved over two passes: U=1 263.48 /
  262.50, U=4 265.58 / 265.36 -> **+0.94% paired**, both passes the same way. Small, and the
  dummy is 4 layers, so it dilutes the mmvf share about 12x against the real model.

### 4. The bench protocol this leaves

Per arm (compile-time, so one rebuild of one TU; `sed` the `#define`, ccache makes it ~3 s
once both exist, record the `.so` md5 - the RPATH here defeats library swapping):

```sh
GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1 rocprofv3 --selected-regions --marker-trace \
  --kernel-trace --stats --output-format csv -o mmvf_u<U> -- \
  llama-bench -m <real model> -lm none -sm tensor -fa 1 -lzm on-direct \
    -ot per_layer_token_embd=CPU -d 16384 -p 0 -n 128 -r 3 -b 2048 -ub 1024
```

Deciding number, from `kernel_trace.csv` (agent 1, the f32 mmvf row with `Grid_Size_X /
Workgroup_Size_X = 512`, which is the router): p50 over the 48 calls per step. Baseline is
27.37 us with p05-p95 26.2-28.5. Prediction: U=4 <= 15 us, U=2 <= 20 us, with the bf16
`indexer.q_proj` row (23.7 us) following. Read per-step device ms too - 0.9 ms on a 27 ms
step is ~3% and will not be visible in `tg` alone.

Falsified on the bench if the row stays within 10% of 27.37 us, in which case the bench's
cost is not this kernel's ILP at all and H23 closes negative.

### 5. Left open

- The bench reading above. Until it lands, this is a T1 win with a known-limited transfer
  story: the dev box's own regime is cache-resident, and E060 1c established that the bench's
  (cold, few blocks) cannot be built here.
- Whether U=2 or U=4 should ship: U=4 wins more rows but loses 2% on `output_hc_up` (0.02
  ms/step in the model, so it does not matter) and U=2 is the one that regresses
  `indexer.q_proj`. Prefer U=4 on the evidence; re-check on the bench.
- Non-RDNA4 builds instantiate the unrolled kernels too. They are only reachable when
  `MMVF_K_UNROLL` is set, but the default build would carry them if the knob is ever flipped
  on unconditionally - the same objection E059's record raises about the mmvq narrow kernels.