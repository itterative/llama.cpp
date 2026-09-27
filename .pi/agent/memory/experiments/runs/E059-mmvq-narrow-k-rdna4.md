# E059 - mmvq on RDNA4 wants fewer warps at narrow K, not more rows per block

Commit `d133df7d4`. Tier **T1+T2**: the mechanism and the tuning grid are dev box (RX 9070,
`test-backend-ops perf` on synthesized shapes), the end-to-end confirmation is the bench box
(4x R9700, real Qwen3.8-Flash-Next Q4_K_M, rocprofv3 decode window).

Raw: [results/E059-mmvq-narrow-k/](../results/E059-mmvq-narrow-k/) (dev-box perf logs, the case
generator, the sweep scripts), [results/user/llama-bench/d133df7d4/](../results/user/llama-bench/d133df7d4/)
(bench-box llama-bench logs and both rocprofv3 stat tables),
[results/user/gguf-dump.log](../results/user/gguf-dump.log) (the real model's tensor census).

## Hypothesis, deciding metric, and how the hypothesis changed

Started from E053's open lead: decode's quantized weight matvecs run at ~115 GB/s against the
R9700's ~640 GB/s, and E054 had already killed "wrong kernel". The first hypothesis was that
`rows_per_block` was too small at narrow K, so each 256-thread block paid a cross-warp reduction
for too few rows. **That was wrong and the first sweep killed it.** The surviving hypothesis is
the opposite: at narrow K the block cannot fill its threads at all, so the fix is fewer warps.

Deciding metric throughout: `avg us/run` for one MUL_MAT_ID op at a fixed shape, from
`test-backend-ops perf`. Effect sizes were 1.1x to 3.8x, far above the ~2% noise floor measured
by re-running identical builds, so no rep count beyond 2-3 was needed at T1.

## The harness, which is the reusable part

`test-export-graph-ops` builds a real pp and tg graph from model metadata and serializes the ops:

```sh
build/bin/test-export-graph-ops -m models/q4exp-4l.gguf -np 1 -c 2048 -ub 2048 -o ops.txt
build/bin/test-backend-ops perf --test-file ops.txt -b ROCm0 -o MUL_MAT_ID -p ',10,1,1\],op_params'
```

The file is plain text (decoder at `tests/test-backend-ops.cpp:11465`, `test_generic_op`):

```
<op> <type> <ne0..3> <n_params> <params...> <num_src> [<src_type> <ne0..3> <nb0..3>]* <name>
```

op 30 = MUL_MAT_ID, type 0 = f32, 6 = q5_0, 8 = q8_0, 12 = q4_K, 13 = q5_K, 14 = q6_K, 26 = i32.
`nb[0] = type_size`, `nb[1] = type_size * (ne0 / blck_size)`, and so on. Type sizes come from the
`static_assert`s in `ggml-common.h`; q5_0 = 22 and q4_K = 144 were confirmed against the real export.

Two properties make this a good instrument. Editing `ne0` synthesizes **the per-card slice a tensor
split produces**, because a TP slice along a contiguous axis is just a smaller contiguous tensor -
so `ffn_down_exps` at k=160, which only exists on a 4-card box, is measurable on one card. And a
synthesized case reproduces the exported one: `down_k640.txt` measured 38.26 us against the real
graph's 38.34, and `n_mats` 64 vs 512 gives 37.92 vs 38.41, so expert count does not enter the timing
(the grid is `m / rows_per_block x n_used`).

Three traps in the tool itself:

- **The GB/s column is meaningless for file-loaded matmuls.** `calculated_bandwidth` is gated on
  `op_flops() == 0` (`tests/test-backend-ops.cpp:1670`) and `test_generic_op` does not override
  `op_flops`, so it prints *allocated* bytes over time. It reported 19516 GB/s for a case that moves
  9.2 MB. Compute bandwidth from the shape by hand.
- `test_generic_op` does not override `reinit_perf_iter` either, so expert ids are **fixed** across
  all ~26k-84k runs and the same working set is re-read. L2 on the RX 9070 is 8 MB (rocminfo), so the
  2.3 MB narrow-K cases are L2-resident. Absolute GB/s from this harness is L2-warm; ratios between
  arms at a fixed shape are unaffected.
- Rebuild is one TU and 34.3 s (`cmake --build build --target ggml-hip -j 8`); CMake globs
  `template-instances/mmq*.cu` and `mmf*.cu` but not mmvq. ccache hashes preprocessed content, so
  `touch` gives a misleading 3.2 s.

## Mechanism, from source

The K loop is `mmvq.cu:699`:

```c
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter)
```

with `blocks_per_iter = vdr * nwarps * warp_size / qi`. So the number of threads that ever enter the
loop body is `blocks_per_row_x * (qi/vdr)`, and it does **not** depend on `rows_per_cuda_block` -
every thread already accumulates all rows of its block in `tmp[ncols_dst][rows_per_cuda_block]`.
That single fact is why the first hypothesis could not work.

`qi` is `QK / (4 * QR)` (`ggml-common.h:103-121`) and `vdr` from `vecdotq.cuh`, so `qi/vdr = 2` for
the block-32 types (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL) and 16 for the K-quants (QI = 32, vdr = 2):

| case | `blocks_per_row_x` | threads entering the loop | of 256 | us |
|---|---|---|---|---|
| down q5_0, k=160 (the 4-card slice) | 5 | 10 | **3.9%** | 35.80 |
| down q5_0, k=640 (one card) | 20 | 40 | 15.6% | 38.29 |
| gate q4_K, k=2560, m=160 | 10 | 160 | 62.5% | 11.95 |

The rest of the block still pays the epilogue: `tmp_shared[nwarps-1][ncols_dst][rows_per_block][warp_size]`
plus `__syncthreads()` plus a `warp_reduce_sum`, to produce 8 output values. At k=160 that is 246 idle
threads and a 7-warp reduction per 8 rows.

The epilogue also gives the hard bound on the lever I first reached for: row `i` is written by
`threadIdx.x == i`, so **`rows_per_block <= warp_size`**.

## Sweep 1 - more rows per block is the wrong lever (negative result)

Raw: [results/E059-mmvq-narrow-k/rows-mult-sweep.log](../results/E059-mmvq-narrow-k/rows-mult-sweep.log).
`MMVQ_RDNA4_ROWS_MULT` scales `rows_per_block = mult * nwarps` under small_k, nwarps = 8, narrow rule
disabled. Single measurements per cell, and the mult=1 column reproduces the baseline three ways
(35.82 / 38.35 / 11.92 here against 35.80 / 38.29 / 11.95 from the exported graph).

| mult | rows | down k=160 | down k=640 | gate m=160 |
|---|---|---|---|---|
| 1 | 8 | 35.82 | 38.35 | 11.92 |
| 2 | 16 | 50.18 | 52.38 | 13.50 |
| 4 | 32 | **164.76** | **179.41** | 16.87 |

Monotonically worse, 4.6x at mult=4. `tmp[ncols_dst][32]` is 32 live floats per thread and
`tmp_shared` grows to 28 KB, so the block spills and occupancy collapses. Recorded because it is the
obvious thing to try and it is very wrong.

## Phase A - the type x K grid

33 cells: 11 whitelisted types x 3 K each, at m=2560 / n_used=10 / n_mats=64. K-quants need k a
multiple of 256, so they get {256, 1024, 2560}; block-32 types get {160, 640, 2560}. Seven arms:
nwarps in {8,4,2,1} x small_k in {on,off}, minus the duplicate (small_k requires nwarps > 1).
Raw logs per arm in [results/E059-mmvq-narrow-k/phaseA/](../results/E059-mmvq-narrow-k/phaseA/),
parser `parse-phaseA.py`.

Best arm per cell:

| arm | cells where best |
|---|---|
| nwarps=1, small_k on | **14 / 33** |
| nwarps=2, small_k on | **11 / 33** |
| nwarps=8, small_k on (the old default) | 6 / 33 |
| nwarps=4 | 2 / 33 |
| either small_k=off arm | **0 / 33** |

That last row is a broad confirmation of E055: small_k on never loses, for any of the 11 types at any
of the 3 K values.

The organizing variable is `blocks_per_row_x = k / blck_size`, not k. At the same k=2560 the block-32
types (kblk=80) prefer nwarps=8 while the K-quants (kblk=10) prefer nwarps=2.

## The rule

RDNA4, `ncols_dst == 1`, type != Q6_K:

| `blocks_per_row_x` | nwarps | rows_per_block |
|---|---|---|
| <= 5 (`MMVQ_RDNA4_NARROW_K_MAX`) | 1 | 1 |
| <= 20 (`MMVQ_RDNA4_MID_K_MAX`) | 2 | 2, small_k on |
| above | table value, 8 | 8, small_k on |

nwarps=2 needs small_k on so `rows_per_block` matches; at nwarps=1 rows is 1 either way.

Implemented as a `narrow_nwarps` int template parameter threaded through `calc_nwarps`, the kernel
template, `__launch_bounds__`, `calc_launch_params`, `mul_mat_vec_q_switch_fusion` and the dispatch
lambda, with a `c_narrow_promoted` guard copied from the existing `c_promoted` one for GB10's
`halve_iters`. `narrow_k` could not be folded into `small_k`: RDNA4's small_k condition is
`blocks_per_row_x < nwarps * blocks_per_iter_1warp` = 128 for block-32 types, so small_k is already
true at kblk=80 where 8 warps genuinely wins, and tightening it would cost 88% there (nw8/small_k-off
= 71.81 us vs nw8/small_k-on = 38.23 at q4_0 k=2560).

**Q6_K is excluded.** It was the only real regression in the grid - 46.19 -> 55.57 us at k=1024, 17%,
reproducible against Phase A's own nwarps=1 arm at 55.39 - and its best warp count is not monotone in
K (1 warp at kblk=1, 8 at kblk=4, 4 at kblk=10). Excluding it returns all three Q6_K cells to baseline
within noise (42.29 / 46.22 / 124.16 against 42.17 / 46.19 / 124.15) and gives up a 1.84x win at
kblk=1 and 1.15x at kblk=10, both at K values that are unusual for a K-quant. Q6_K turns out to be
heavily used in the real model (see below), so the exclusion protects real work.

### Grid outcome, all 33 cells against the old default

| kblk | ratio range | note |
|---|---|---|
| 1 | **2.33x - 3.84x** | iq4_xs 75.48 -> 19.68 us |
| 4 | 1.20x - 2.74x | Q6_K excluded, back to baseline |
| 5 | **1.59x - 2.13x** | the real 4-card `ffn_down_exps`; all six block-32 types |
| 10 | 1.33x - 1.74x | q2_K, iq4_xs; q4_K and q5_K neutral on the dev box |
| 20 | **1.14x - 1.67x** | the real one-card `ffn_down_exps`; all six |
| 80 | 0.96x - 1.00x | the rule returns 0 here, so this row is inert by construction |

22 of 33 cells win by more than 5%. The kblk=80 row also supplies the noise floor: two runs of
*identical* code differ by up to 3.9% there (q4_1 36.95 vs 37.84), which is why the two sub-0.98x cells
in that row are not regressions.

## Correctness

- Golden gate **267035.3875**, bit-identical, checked three times: after the config-header extraction
  alone, after adding narrow_k, and after the Q6_K exclusion plus the final restore.
- `test-backend-ops test -b ROCm0 -o MUL_MAT` and `-o MUL_MAT_ID`: 2/2 backends passed at each stage.
- The extraction step (`mmvq-config-rdna4.cuh`) was verified separately before any tuning: 4/4 perf
  cases within noise (gate m=640 22.59 -> 22.46, down k=640 38.26 -> 38.21, down k=160 35.78 -> 35.82,
  down k=160 with small_k off 65.93 -> 65.93).
- No reassociation anywhere: `narrow_nwarps` changes the block shape only. Each thread still walks the
  same K blocks in the same order for its rows.

## T1 end to end, dev box

`llama-bench -m <dummy> -ngl 99 -p 0 -n 128 -r 3 -fa 1 -lzm on-direct -sm none`,
`Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1`, two interleaved cycles. Arms differ by rebuilding the
two macros to 0 - **`GGML_CUDA_MMVQ_RDNA4_SMALL_K=0` does not disable this rule**, `narrow_k_nwarps`
never consults it, so using the env var as a baseline would silently compare treatment to treatment.
`.so` md5 per arm confirmed distinct and reproduced across cycles (base `5e8c0529`, treat `300a719e`).

| model | base t/s | treat t/s | delta |
|---|---|---|---|
| q4exp-4l (512 experts, 4 layers) | 251.88, 249.90 | 268.28, 267.06 | **+6.7%** |
| q4exp-48l-12qsa | 39.28, 39.14 | 43.80, 43.85 | **+11.8%** |

The 48-layer model gains more, the same pattern E055 saw (+4.07% on 4l vs +7.4% on 48l): this is
per-expert-matmul work, so it scales with layer count.

RPATH on this box defeats `.so` swapping for an interleaved A/B - `LD_LIBRARY_PATH` is ignored and
`LD_PRELOAD` fails, and mixing libs from two trees core-dumps - so arms have to be rebuilt. ccache
makes each rebuild ~3 s once both variants exist, so four arm switches cost ~15 s.

## T2 confirmation, bench box

`results/user/llama-bench/d133df7d4/`. Baseline built at the parent commit `10fa6648e`, optimized at
`d133df7d4`; both arms
`GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_BF16=nccl GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1`,
`-sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -d 4096,16384,40960,131072 -p 0 -n 128 -r 3 -b 2048 -ub 1024`.

| depth | baseline | optimized | delta |
|---|---|---|---|
| d4096 | 36.05 +/- 2.86 | 37.59 +/- 3.16 | **+4.27%** |
| d16384 | 35.44 +/- 2.66 | 37.09 +/- 3.00 | **+4.66%** |
| d40960 | 34.62 +/- 2.56 | 36.09 +/- 2.78 | **+4.25%** |
| d131072 | 31.59 +/- 2.09 | 32.76 +/- 2.29 | **+3.70%** |

Flat in depth, as expected for a per-matmul change and unlike the pool's, so it is additive on H9 and
on E055's small_k.

**Protocol note.** The effect is under 5% and the arms were sequential `-r 3`, not the alternating
`-r >= 10` the protocol requires at that size. What substitutes for it is the mechanism check the
protocol prefers over more reps: an independent device-time accounting that predicts the wall gain.
It does: 1.35 ms of device time removed per card per step against 1.13-1.26 ms of wall gained, so
84-93% of it is exposed.

### The trace accounting

Both rocprofv3 stat tables are **complete** (the `%` column sums to 100.02 and 100.00), unlike E053's,
which was truncated at 82.98 - so these totals can be used directly.

Controls, identical in both arms, which is what makes the comparison trustworthy:

| quantity | baseline | optimized |
|---|---|---|
| `mul_mat_vec_q` calls | 799,260 | 799,260 |
| `quantize_q8_1` calls | 799,260 | 799,260 |
| `ar_oneshot` calls | 147,840 | 147,840 |

799,260 / 4 cards / 385 steps = **519 matvecs per card per step**, independently reproducing E053's
count, and 147,840 / 4 / 385 = 96 collectives per card per step, reproducing its 2-per-layer figure.
One `quantize_q8_1` per matvec, which is the activation re-quantize E053 flagged.

**The new path ran** - the 6th template argument is visible in the demangled names and exists only in
the optimized arm:

| kernel | per card/step | base us | opt us | ratio |
|---|---|---|---|---|
| Q5_0, small_k, narrow=2 | 102 | 15.36 | **7.39** | **2.08x** |
| Q5_0, narrow=1 | 18 | 15.36 | 10.05 | 1.53x |
| Q4_K fused gate+up, small_k, narrow=2 | 48 | 14.95 | **11.00** | **1.36x** |
| Q8_0, small_k, narrow=2 | 54 | 9.09 | **5.89** | **1.54x** |
| Q5_K, small_k, narrow=2 | 6 | 12.77 | 10.93 | 1.17x |
| Q4_K, small_k, narrow=2 | 59 | 7.88 | 7.10 | 1.11x |
| Q8_0, narrow=1 | 18 | 9.09 | 9.04 | 1.01x |
| Q6_K, narrow=0 (excluded) | 40 | 17.96 | 17.66 | untouched |
| Q6_K fused, narrow=0 | 24 | 6.98 | 6.88 | untouched |
| Q4_K, small_k=0, narrow=0 | 96 | 4.79 | 4.63 | untouched |

Totals from `kernel_stats.csv`, summed over all four agents:

| | baseline | optimized | ratio | saved |
|---|---|---|---|---|
| `mul_mat_vec_q` | 8794.8 ms | 6702.9 ms | **1.312x** | 2091.9 ms |
| all listed device time | 24,663.3 ms | 22,286.4 ms | 1.107x | 2376.9 ms |

The authoritative version is straight off `kernel_trace.csv` filtered to `Agent 1` and summed as
`End_Timestamp - Start_Timestamp`: **15.99 -> 14.64 ms per card per step, saving 1.35 ms (-8.4%)**,
against the 16.02 -> 14.47 / 1.55 ms the aggregate table implies. The ~1% gap is rows the stats
aggregate counts that the per-agent filter does not; use the raw-trace figure.

**Reconciliation with the wall.** tg improved by 1.13 / 1.26 / 1.18 / 1.13 ms per token at the four
depths against 1.35 ms per card per step of device time removed, so **84 / 93 / 87 / 84% of the saving
reaches the wall**. Two independent measurements agreeing is the stronger half of this record.

**Do not scope device time to marker ranges.** An earlier version of this section did, and got 8.31 ms
per card per step instead of 14.64. `phase:decode` is a *host* span around an async graph launch, so
device execution spills past its end: of Agent 1's 1,392,129 kernels, 810,565 (3,303 ms) fall inside the
385 markers and 581,564 (2,333 ms) fall in the gaps between them - the same kernels (`mul_mat_vec_q`
83,841, `ar_oneshot` 15,626, `mul_mat_vec_f` 49,506), just executing late. The median inter-marker gap is
13.1 ms and carries ~6.1 ms of device work. Marker duration is therefore host launch time, not token
wall: the profiled token period is ~36.7 ms (23.6 ms marker + 13.1 ms gap) against 26.6 ms unprofiled, so
~10 ms/token is interception cost at 3,616 dispatches per card per step. Use device totals over the whole
trace, and the unprofiled wall.

The Q6_K exclusion is validated on the real model: 64 calls per card per step across `attn_qkv`,
`attn_q`, the shared-expert gate/up and `output.weight`, all untouched.

### Where decode device time goes now (complete table, optimized arm)

| group | ms | share | per card/step |
|---|---|---|---|
| `mul_mat_vec_q` | 6702.9 | **30.1%** | 4.35 ms |
| `ar_oneshot` (collective) | 5181.8 | **23.3%** | 3.36 ms |
| `mul_mat_vec_f` (f32/bf16) | 4014.9 | **18.0%** | 2.61 ms |
| `quantize_q8_1` | 893.3 | 4.0% | 0.58 ms |

Device busy is 14.64 ms per card per step (raw trace, Agent 1, whole trace over 385 steps). Against the
reported 26.6 ms/token wall at d4096 that is **55%**, so ~45% of the wall is host-side or gaps and stays
deliberately parked. `meta:subgraph`'s 21.2 ms/step is *not* additive cost - it is a host span that
overlaps device execution - which is why it is not comparable to E058's 5.4 ms/token.

## Re-analysis of the raw traces (after the first commit)

The first pass read only `kernel_stats.csv`. The profile directory holds four more file kinds, and
`kernel_trace.csv` carries per-dispatch launch geometry and resource use: `Workgroup_Size_X/Y`,
`Grid_Size_X/Y`, `VGPR_Count`, `SGPR_Count`, `Scratch_Size`, `LDS_Block_Size`. Two traps: the name field
contains commas so positional `$N` indexing is wrong and `$(NF-k)` from the end is required, and the block
is **2D** - `Workgroup_Size_X` alone reads as 32 for every kernel and hides nwarps, which is in `_Y`.
All four GPUs are in the **one** file (`Agent 1..4`, 1,392,129 dispatches each, common monotonic clock),
which makes cross-rank skew measurable. Use duckdb, not awk: `read_csv(..., header=true, quote='"')` over
the 1.8 GB trace answers in ~0.7 s and parses the quoted names correctly.

**The narrow rule fired exactly as designed**, now from launch geometry rather than template args:

| template `narrow` | observed block | types seen |
|---|---|---|
| 1 | 32x1 = **32 threads** | q5_0, q8_0 |
| 2 | 32x2 = **64 threads** | q5_0, q8_0, q4_K, q5_K |
| 0 | 32x8 = **256 threads** | q8_0, q4_K, **q6_K** |

`Workgroup_Size_X = 32` everywhere confirms wave32 on gfx1201, so the "10 of 256 threads" figure in the
mechanism section is right. **q6_K appears only ever at narrow=0**, the exclusion verified empirically
rather than inferred. `Scratch_Size = 0` on every mmvq row and `LDS_Block_Size <= 7168` of the 64 KB
available, so no shipped config spills; VGPR use is 16-56.

Dispatch count is **identical in both arms, 1,392,129 per card**, alongside the 799,260 matvec control:
the change alters block shape and nothing else about the work submitted.

### Three pathological host spans, a 17 s gap, and what is not yet explained

`phase:decode` is not unimodal. At identical step indices in both arms:

| step | baseline | optimized |
|---|---|---|
| 1 | 989 ms | 1040 ms |
| 3 | 526 ms | 527 ms |
| 258 | 675 ms | 670 ms |

385 steps = 3 x 128 + 1, so this is the `-r 3` tg128 run; the matvec count confirms it independently
(799,260 / 4 cards / 385 = 519 per card per step). Those three spans are 2.1 s of the 11.3 s the markers
cover, and they cancel in this A/B only because they hit both arms. Step 1's is a single 598 ms
`meta:subgraph`; step 258's has no matching subgraph outlier. Median marker span is 22.6 / 22.1 ms and
p95 is 27.2 / 26.2, so the rest is tight.

The gaps *between* markers are a separate population and total 22.1 s: a **17.0 s gap after step 2**,
150 ms before step 130 and 129 ms before step 258 (the rep boundaries), and a 13.1 ms median elsewhere.
The 17 s gap is not decode work and is unexplained.

Hypothesis for the long spans, not measured: first-touch page faults on the 133 MB n-gram table, i.e.
E058's compulsory misses arriving at once. **Whether these also occur unprofiled is not established** -
every figure here comes from the rocprofv3 run - so the earlier claim that all bench-box t/s for this
model carry ~5.5 ms/token of inflation is unverified and is a question, not a fact.

### The host slack is dispatch count, not one slow range

Agent 1 kernel durations: p50 **1.4 us**, p90 8.4, p99 27.7, p99.9 224.5, max 15.66 ms. **63% of all
kernels are under 2 us** - about 2,285 of the 3,616 dispatches per card per step - holding 18% of device
time. So the ~45% host slack is per-dispatch overhead from kernel *count*, not a mysterious range. The
lever is fewer launches, not faster kernels. Per card per step the dispatch population is `mul_mat_vec_q`
519, `mul_mat_vec_f` 302, `unary_op_kernel` 294, `k_bin_bcast` 257, `rms_norm_f32` 244, `ar_oneshot` /
`dsv4_hc_pre_f32` / `dsv4_hc_post_f32` 96 each, `unary_gated_op_kernel` 79 - so ~926 elementwise and norm
launches against 519 matvecs.

`marker_api_stats.csv` quantifies the host side directly (nested ranges, so shares do not sum):
`phase:decode` 29.24 ms/step mean, `graph:compute` 27.74 ms = 94.9% of it, `meta:subgraph` 37,345 calls
= **97 per step** at 21.2 ms/step, `meta:allreduce` 36,960 = 96 per step at **1.71 ms of host time
against 3.36 ms of `ar_oneshot` device time**, so the collective is roughly half host-bound. E058's input
path on the bench box: `graph:set_inputs` 858 us/token, `input:lazy_gather` 716 us, `lazy:prefetch`
182 us, and `input:lazy_h2d` / `lazy_staging` / `lazy:sort` all under 3 us, which closes E058's open
question about those three.

`agent_info.csv` gives the R9700 limits for occupancy arguments: 64 CU, 2 SIMD/CU, 16 max waves/SIMD,
64 KB LDS, wave32 and wave64 both available, `Workgroup_Max_Size` 1024.

## What the GGUF dump corrected

`results/user/gguf-dump.log`, shard 1 of 4, 64 tensors.

- **`ffn_down_exps` is Q5_0 in some layers and Q8_0 in others**, both `[640, 2560, 512]`. E053 recorded
  it as uniformly Q5_0 and flagged its own byte table as derived rather than measured. Both variants
  sit at kblk=5 per card, so both take the narrow=1 band, and both types were in the grid.
- **`ffn_gate_exps` and `ffn_up_exps` are separate Q4_K tensors** `[2560, 640, 512]` in the file, fused
  in the graph: the `has_fusion=true` Q4_K row is exactly 48 per card per step, one per layer.
- **E058's 110-byte n-gram row is confirmed and its decomposition corrected.**
  `per_layer_token_embd.weight` is Q5_0 `[160, 320001536]`, so a row is 160 elements = 5 Q5_0 blocks
  of 32 x 22 B = **110 B exactly**. E058 recorded "one Q3_K block of 256 elements": right byte count,
  wrong structure. The row count is 320,001,536, not the 20M x 16 = 320M E058 derived.
- Q6_K appears on `attn_qkv` `[2560, 10240]`, `attn_q` in some layers, some shared-expert gate/up, and
  `output.weight` `[2560, 248320]`. Q5_K on `attn_output` `[6144, 2560]`, which splits to k=1536 per
  card = kblk 6, just inside the mid band.

## Caveats

- Dev-box absolute GB/s is L2-warm (8 MB L2, fixed expert ids, 2.3-9.2 MB working sets). The bench box
  is DRAM-cold and confirmed the win anyway, which is the answer to the question the dev box could not
  settle: fewer warps does not cost latency hiding here.
- The bench arms were sequential, not alternating, at `-r 3` with per-depth spread of 6-8%. No single
  depth is significant alone. What carries it is that all four move the same way by 3.7-4.7% and the
  device-time accounting independently predicts the size.
- **Never scope device time to ROCTX marker ranges.** Graph launch is async and 42% of device work
  executes between markers, so a marker-scoped per-step figure understates it by that much. Marker
  duration is host launch time, not token wall.
- Absolute profiled throughput is not comparable to unprofiled: ~36.7 ms/token under rocprofv3 against
  26.6 ms without. Ratios between arms inside one trace are safe.
- The three long host spans and the 17 s gap are observed in the profiled run only. Whether they occur
  unprofiled, and so whether bench-box t/s for this model is inflated, is unestablished.
- Non-RDNA4 builds instantiate the narrow kernels they will never launch, because `c_narrow_promoted`
  has to hardcode `MMVQ_PARAMETERS_RDNA4` - there is no host-side constexpr for the device table, which
  is exactly why the existing `c_promoted` hardcodes GB10. Roughly 44 extra instantiations for the 11
  whitelisted types at `ncols_dst == 1`. An upstream reviewer will raise this.
- RDNA3 is still unmeasured, as in E055. Its own stricter whitelist is untouched, and `narrow_k_nwarps`
  gates on `table_id == MMVQ_PARAMETERS_RDNA4`.

## Closed and left open

Closed: E053's "the ~115 GB/s has to be explained inside mmvq" lead, and backlog L3. The framing that
survives is narrower than "a fifth of peak bandwidth" - the kernel was never bandwidth-limited, it could
not fill its own blocks, and the kblk=5 case reaches 64 GB/s while fully L2-resident.

Still open, re-ranked by the complete table above and by the trace re-analysis:

1. **The three long host spans and the 17 s gap** - 989 / 526 / 675 ms at steps 1, 3 and 258, equal in
   both arms, 19% of the marker-covered window, plus a 17.0 s gap after step 2. All observed under
   rocprofv3 only, so whether they occur unprofiled is the first thing to establish. If they are
   first-touch faults on the 133 MB n-gram table this connects directly to E058's WILLNEED work, which
   already buys +4.2% by prefaulting 16 rows per step.
2. `ar_oneshot` at 23.3% is now within 7 points of all quantized matvecs combined, and is **roughly half
   host-bound**: 1.71 ms of `meta:allreduce` host time against 3.36 ms of device time per card per step.
   Cross-rank skew is now measurable too, since all four agents share one trace file on a common clock -
   that is the direct test of E044's "the LL spin may be waiting out a peer that has not arrived".
3. `mul_mat_vec_f` at 18.0% / 302 calls per card per step. E053 estimated this group at 6.1% from a
   truncated table; it is three times that and no change here touches it.
4. **Host dispatch count**, not host dispatch cost: 3,616 dispatches per card per step, 63% of them under
   2 us, and ~45% of the wall is host-side or gaps. `meta:subgraph` is 97 spans per step *enclosing*
   those dispatches, so its 21.2 ms/step is not additive cost and the item is not "5.4 ms of dispatch" as
   first written. The lever is fewer launches. Parked by the user's call.
5. `quantize_q8_1` at 4.0%, one launch per matvec, 799,260 of them. `mmvq.cu:1503-1509` does an
   unconditional fresh `ggml_cuda_pool_alloc` plus quantize per call with no memoization on `src1`.
   Smaller in relative terms now, but the mechanism is still there.
6. Q8_0 at kblk=80 wants 1 warp by 1.68x (118.00 -> 70.27 us on the dev box), the one wide-K cell that
   breaks the pattern, and Q8_0 is 90 calls per card per step in the real model. Outside this rule by
   design; worth its own look.
