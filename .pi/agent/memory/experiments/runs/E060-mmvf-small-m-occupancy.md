# E060 - does mmvf lose its small-m matvecs to the block-size heuristic

- date: 2026-09-27
- machine: dev-rx9070-16g (T1), bench-4x-r9700-32g for the end-to-end effect
- tier: T1, then a T2 confirmation of the winner only
- status: done
- parent: E059
- commit: `251b517d7` + the `GGML_CUDA_MMVF_BLOCK_SIZE` knob (mmvf.cu, default off)
- build: as E059 (Release, HIP, gfx1201); instrument is `GGML_CUDA_MMVF_BLOCK_SIZE`
- model: op-level shapes from the real Q4_K_M tensor census (no weights needed)

## hypothesis

`launch_mul_mat_vec_f_cuda` picks `block_size_best` by minimising the number of serial K
iterations, `niter = (ncols + 2*block_size - 1) / (2*block_size)`, and nothing else
(`mmvf.cu:437-455`). The kernel is one row per block with K split across the block
(`row = blockIdx.x`, `for (i = tid; i < ncols/2; i += block_size)`), so a bigger block
buys fewer iterations per lane and pays a shared-memory reduction plus two
`__syncthreads()` per row. For a small-m 1-column matvec that trade is wrong in both
directions at once: the grid cannot exceed `m` blocks (so m=4 means four blocks), and each
of those blocks ends with a block-wide reduction over 8 warps.

E059's decode window says this is where `mul_mat_vec_f`'s 2.61 ms/step (18.0% of device
time per card) sits, and the rows are 4-190 GB/s while moving 8 KB to 5 MB:

| tensor (per-card shape k x m) | type | split | calls/step/card | us/call | ms/step | bytes/call | GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ffn_gate_inp` router, 2560 x 512 | f32 | mirrored | 48 | 27.61 | **1.33** | 5.24 MB | **190** |
| `indexer.q_proj`, 2560 x 512 | bf16 | mirrored | 12 | 23.79 | 0.29 | 2.62 MB | **110** |
| `indexer.k_proj`, 2560 x 128 | bf16 | mirrored | 12 | 2.58 | 0.03 | 655 KB | 254 |
| `hc_*_inject`, 10240 x 4 | bf16 | mirrored | 96 | 5.90 | 0.57 | 82 KB | **14** |
| `ssm_alpha`/`ssm_beta`, 2560 x 12 | f32 | axis 1 | 72 | 2.49 | 0.18 | 123 KB | **49** |
| (m = 1 row) | f32 | - | 48 | 2.31 | 0.11 | 10 KB | **4** |
| `output_hc_down`, 10240 x 320 | bf16 | mirrored | 1 | 15.33 | 0.02 | 6.55 MB | 427 |
| `output_hc_up`, 320 x 10240 | bf16 | mirrored | 1 | 22.47 | 0.02 | 6.55 MB | 291 |

The last two rows are the internal control: same kernel, same file, m large, already at
291-427 GB/s. The heuristic also reproduces exactly in the trace (`Workgroup_Size_X` is
160 for k=320 and 256 for k=10240, which is what the niter loop computes), so the knob
tests the heuristic and not a misreading of the trace.

## prediction

Deciding metric: `avg us/run` per case from `test-backend-ops perf --test-file`, at the
per-card shapes above, `ncols_dst == 1`, arms differing only in forced block size.

| arm | block size | what it changes |
| --- | --- | --- |
| A0 | heuristic (control) | 160/256 as above |
| A1 | 32 | one warp per row, warp-shuffle reduction, no shared memory and no barrier |
| A2 | 64 | two warps per row |
| A3 | 128 | four warps per row |

Expected, if the hypothesis holds: A1/A2 beat A0 by at least 1.5x on the four rows with
m <= 512 (router 27.6 -> <= 16 us, indexer.q 23.8 -> <= 14, ssm 2.49 -> <= 1.6, m=1
2.31 -> <= 1.5), with the biggest relative wins where the grid is smallest. The two large-m
control rows should be flat within noise, and the m=10240 and m=320 rows must not regress
by more than 5% or the change is not shippable as a blanket rule.

Aggregate bound: the small-m rows are ~2.4 of the 2.61 ms/step. A 2x win on them is
1.2 ms/step of 14.47 (8.3% of device time), which at E059's 73-93% exposure is 3-4% of
`tg` - under the 5% bar, so the T2 confirmation is the decode-window device accounting
(controlled by identical dispatch counts), never a single `tg` reading.

Falsified if every arm is within 10% of A0 on all six small-m rows. The negative is still
worth its slot: it moves the target from "launch geometry" to "launch count", i.e. fuse
the 96 `hc_*_inject` and 72 `ssm_*` matvecs, and it says the 5.90 us for an 82 KB matvec is
dispatch and tail, not occupancy.

## conditions

Dev box, per `PROTOCOL.md` 5.1 and the E059 harness:

- `export LD_LIBRARY_PATH=$PWD/build/bin`, device filter `ROCm0` (a filter that matches
  nothing passes silently), two runs per cell, and the A0 cell repeated at the end of the
  sweep to measure the day's noise floor at these shapes.
- One build per arm (`mmvf.cu` is one TU, ~35 s cold, ~3 s with ccache); the `.so` md5 per
  arm gets recorded because the RPATH here defeats `LD_LIBRARY_PATH` swapping at runtime.
- Correctness: the knob changes the K-split width per row, so the sum order changes and the
  result is **not** bit-exact by construction. Gate is `test-backend-ops test -b ROCm0 -o
  MUL_MAT` (2/2 backends) plus an explained diff: E018 golden corpus `PPL` and
  `scripts/compare-logprobs.py` max/mean abs delta recorded, not required to be zero. A
  change that moves the numbers by more than the f32 summation noise is a bug, not a
  tuning result.
- Also record `-o MUL_MAT_ID` on the four E059 shapes, because the same heuristic is used
  for the id path and a regression there would cost more than the win.

## not in scope

- No new kernel, no change to `vec_dot`, no numerics review beyond the diff gate.
- No bench-box run until a T1 winner exists; no `tg`-only verdict at any point.
- No change to non-RDNA4 paths: the knob is env-gated and off by default.

## results

Verdict: **negative**. The hypothesis is refuted in the direction it predicted (a smaller
block is *worse* at small m), and the T1 instrument turned out to be blind for the two rows
that carry the bytes. Both are worth keeping.

Commit `251b517d7` plus the `GGML_CUDA_MMVF_BLOCK_SIZE` knob. One binary for every arm (the
knob is host-side), `.so` md5 `a29534640576633ae17449c12d832029`. One TU, 20 s build. Raw:
[results/E060-mmvf-occupancy](../results/E060-mmvf-occupancy/) - 7 op-level arms, 6
end-to-end arms, 3 PPL arms, the case generator, the sweep and the parser; the raw `.log`
files are untracked, as in E059, because `.gitignore` has `*.log` - the sweep and the PPL commands
regenerate them in ~2 minutes).

### 1. The op-level harness measures exactly these rows cache-resident, by 3.3-3.8x

`test_generic_op` re-reads one working set: the router case ran 143,858 times against the
same 5.24 MB tensor, which fits the 8 MB L2. The bench trace reads that tensor once per
step behind ~1.5 GB of other traffic. Same shapes, both instruments:

| per-card shape | dev harness us | bench device us | ratio | weight |
| --- | --- | --- | --- | --- |
| f32 2560 x 512 (router) | 7.33 | 27.37 | **0.27** | 5.24 MB |
| bf16 2560 x 512 (indexer.q_proj) | 7.89 | 23.69 | **0.33** | 2.62 MB |
| bf16 10240 x 320 (output_hc_down) | 11.36 | 15.21 | 0.75 | 6.55 MB |
| bf16 320 x 10240 (output_hc_up) | 25.75 | 22.06 | 1.17 | 6.55 MB |
| bf16 10240 x 4 (hc_*_inject) | 6.15 | 5.79 | 1.06 | 82 KB |
| bf16 2560 x 128 (indexer.k_proj) | 3.77 | 2.52 | 1.50 | 655 KB |
| f32 2560 x 12 (ssm_alpha/beta) | 3.57 | 2.46 | 1.45 | 123 KB |
| f32 2560 x 1 | 3.55 | 2.22 | 1.60 | 10 KB |

So the cache-sized rows are measured 3.3-3.8x too fast, and the tiny-m rows 1.45-1.60x too
slow (this box has a higher launch/tail floor). The two rows that matter - router plus
indexer.q_proj, 1.62 of the group's 2.61 ms/step - are both in the first category. Every
arm below is therefore cache-warm data, and E059's caveat about this harness is now
measured rather than assumed.

### 1b. The cache that matters is the 64 MB Infinity Cache, not the 8 MB L2

Sweeping m at k = 2560 f32 (one row per block, so the grid grows with m) brackets the
level: the effective rate keeps *rising* to 1396 GB/s, well past the 640 GB/s DRAM peak,
and then falls off a cliff at 64 MB.

| m | weight | us | effective GB/s |
| --- | --- | --- | --- |
| 1024 | 10.5 MB | 11.04 | 950 |
| 2048 | 21.0 MB | 18.07 | 1161 |
| 4096 | 41.9 MB | 31.59 | 1328 |
| 5120 | 52.4 MB | 38.23 | 1371 |
| 6144 | 62.9 MB | 45.07 | **1396** |
| 7168 | 73.4 MB | 110.78 | 663 |
| 8192 | 83.9 MB | 134.42 | 624 |

So the trap is wider than "small cases": *every* per-tensor read in the model is under
64 MB (the largest, the Q6_K lm head at 130 MB/card, is the one row whose op-level rate
agreed with the bench - 577 GB/s there, DRAM-bound by construction). The earlier note about
this box's "8 MB L2" is the L2; the level that decides whether an op-level number means
anything is the MALL.

### 1c. Can the harness be made cold? Three walls, and what they leave

Tried, in order of increasing fidelity:

1. **Grow m until the weight exceeds the cache.** Works for the cache bracket, but it
   changes the shape: m = 512 with a cold weight cannot be built, because the weight is
   `k * m * 4` and m is fixed by the model.
2. **Batch dim on ne2**, so one node reads B distinct slices (the only lever the case file
   has). This does read cold (B = 16 x 5.24 = 84 MB) but the grid becomes (m, B) blocks, so
   the B slices run concurrently and land in the DRAM-bound regime: 8.34 us per slice at
   629 GB/s, against the bench's 27.37 us for the same slice. Useful as a *floor*, not as a
   reproduction.
3. **Rotate the source pointer between iterations.** `test_case::reinit_perf_iter` is called
   once per `ggml_backend_graph_compute`, while `eval_perf` adds the *same node* to the graph
   `min(avail, 32 GB / op_size) + 1` times per call - 6103 repeats for the router - so a
   rotation moves the address once per 6103 dispatches. Dead end by construction.

The faithful instrument for a sub-64 MB tensor is therefore a model at streaming scale:
the bench decode window (already built), or a model-level dev-box run, which needs a
per-kernel timing path the dev box does not have (`rocprofv3` is absent; `GGML_PROF_REGIONS`
spans are host time and must not be read as device time).

### 1d. What the cold probes did settle: the router is bytes-in-flight bound

The probes keep the bench's own geometry (one row per block, 512 blocks) and grow k, which
adds independent iterations per thread:

| k, m = 512 | weight | us | GB/s |
| --- | --- | --- | --- |
| 2560 | 5.2 MB | 7.19 | 729 (cache) |
| 8192 | 16.8 MB | 15.26 | 1099 (cache) |
| 16384 | 33.5 MB | 23.89 | 1405 (cache) |
| 32768 | 67 MB | 51.42 | 1305 (cache edge) |
| 65536 | 134 MB | 219.31 | **612 (DRAM)** |

512 blocks *can* saturate DRAM (612 GB/s at k = 65536) - the block-count theory of the
router's 27.37 us is dead. E061 tests the resulting lever (more independent loads per thread)
and finds the dev-box win; the bench reading is still the deciding number. Note the screening
consequence: the cache-resident m = 512 case
(7.19 us) is latency-bound on the *same* limit - 5 loads per thread - so a change that gives
each thread more independent work should move it here, before the bench is asked to confirm
the cold number. What the router lacks is independent work per thread: 5 float2
loads at k = 2560 against 128 iterations at k = 65536, with the same bytes and the same
grid. That is why the cold block-size sweep is inert where it matters: in the DRAM-bound cases
bs32/bs64/bs128/bs256 move it by 3% or less, with one 13% outlier at bs128 (k = 65536:
219.3 / 219.3 / 233.3 / 248.3 / 219.3 us; the B=16 router batch stays inside 2%), while the
same knob still wrecks the tiny rows (inject 6.37 -> 16.95 us at bs64).

### 2. The hypothesis is refuted where the instrument can see it

`test-backend-ops perf`, `avg us/run`, two controls (bs0 and bs0r, same setting twice):

| case | bs0 | bs0r | bs32 | bs64 | bs96 | bs128 | bs160 | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 320 x 10240 | 25.75 | 26.33 | 26.15 | **15.56** | 19.30 | 22.84 | 26.10 | bs64, 1.65x |
| f32 2560 x 512 | 7.33 | 7.64 | 7.51 | 6.53 | **6.00** | 8.01 | 7.53 | bs96, 1.22x |
| bf16 2560 x 512 | 7.89 | 7.92 | 7.91 | 6.36 | **5.65** | 7.29 | 7.44 | bs96, 1.40x |
| bf16 10240 x 320 | 11.36 | 11.19 | 11.18 | 16.50 | 12.25 | 9.83 | **8.81** | bs160, 1.29x |
| bf16 2560 x 128 | **3.77** | 3.79 | 3.80 | 6.29 | 5.35 | 4.64 | 4.28 | heuristic |
| f32 2560 x 12 | **3.57** | 3.59 | 3.55 | 6.24 | 5.17 | 4.48 | 4.15 | tie |
| bf16 10240 x 4 | **6.15** | 6.18 | 6.25 | 16.81 | 12.30 | 9.89 | 8.40 | heuristic |
| f32 2560 x 1 | 3.55 | **3.54** | 3.54 | 6.18 | 5.14 | 4.42 | 4.09 | tie |

Noise floor at these shapes is 0.4-2.3% (bs0 vs bs0r), so the columns are clean.

- **Smaller block is worse, not better, at small m.** bs32 sits on the heuristic; bs64 is
  1.7-2.7x worse on the m = 1..128 rows. The heuristic maximises threads per row to cut the
  serial iteration count `ncols/2/block_size`, and that count, not the block reduction, is
  what bounds these rows. First prediction falsified.
- **The control rows moved too.** bs64 is 1.65x on 320 x 10240 and bs160 1.29x on 10240 x
  320, so "large m is already at the bound" was wrong as well. Second prediction falsified.
- The m = 512 rows do win 1.22x / 1.40x at bs96, but those are the L2-warm rows, so this is
  not evidence about the bench. If it transfers, 1.62 ms/step becomes ~1.25.
- The best value is per-shape: bs96 wins 2 rows and loses 4, bs64 wins 1 and loses 4.

### 3. End-to-end, DRAM-cold: a blanket override loses

Dummy `q4exp-4l`, `-ngl 99 -sm none -fa 1 -lzm on-direct -d 4096 -n 128 -r 3`, arms
interleaved as 0/96/64 twice, 12.76 GB of weights so each step is cold:

| arm | pass 1 | pass 2 | mean | delta |
| --- | --- | --- | --- | --- |
| bs0 (heuristic) | 262.65 | 262.32 | 262.5 | - |
| bs96 | 258.78 | 258.23 | 258.5 | **-1.5%** |
| bs64 | 252.10 | 251.99 | 252.0 | **-4.0%** |

Both lose, and in the order the op-level mix predicts: bs64's damage is concentrated in the
rows it triples (inject 96/step, indexer.k, ssm). A blanket block-size override is **not
shippable**; the heuristic is already near-optimal in aggregate, even though it is 1.2-1.7x
off the per-row optimum on four rows.

### 4. Correctness

`GGML_CUDA_MMVF_BLOCK_SIZE=96 test-backend-ops test -b ROCm0 -o MUL_MAT`: **1297/1297
passed, 2/2 backends**. Golden corpus PPL: 263113.6984 (bs0), 263113.3567 (bs96, -1.3e-6
relative), 263113.8181 (bs64, +4.6e-7). Non-bit-exact by construction - the block size sets
the K-split width, so the sum order changes - and the drift is at f32 summation noise.
`compare-logprobs.py` was not run: the PPL delta is already at the 1e-6 level and the op
gate covers the touched kernel.

The pre-registered `-o MUL_MAT_ID` guard is vacuous for this knob: it is read only in
`launch_mul_mat_vec_f_cuda`, so the id path (`..._vec_q`) cannot see it. E059's MUL_MAT_ID
gate stands.

### 5. Harness trap worth keeping

`test_generic_op` parses the case fields with `operator>>` into integers, so a field written
in scientific notation is silently truncated to its mantissa: `5.24288e+06` becomes `5`,
the strides collapse, and the case aborts in rocBLAS with `invalid configuration argument`
(the generator here prints integers only). Cost: one crash loop.

### 6. What is left

- The mmvf group is 2.61 ms/step (18% of device time): router 1.33, inject 0.57, indexer.q
  0.29, everything else ~0.42. The block-size lever is dead in both regimes - per-row optima
exist only cache-warm, and the cold sweep is inert. What survives is the mechanism:
  the router row is DRAM-latency-bound at ~5 independent loads per thread (190 GB/s), and
  the same bytes at the same 512-block grid reach 612 GB/s once each thread has enough
  independent work. The cold batched proxy puts the prize at 8.34 us per slice against
  27.37, i.e. **~0.9 ms/step (6% of device time) if 48 routers could be issued with that
  much concurrency**, and the lever is more independent work per thread, not a different
  block size: register-blocked rows per thread, float4 instead of float2 loads, or a K-split
  across 2-4 blocks.
- The other 200+ calls sit on a ~3.5 us launch/tail floor while moving 10-655 KB, so the
  opposite change is also live: fewer launches (fuse the 96 `hc_*_inject` and 72 `ssm_*`
  calls per step, H24).
- Instrument lesson for the rest of the project: an op-level number for a tensor under
  64 MB describes the MALL, not DRAM, and the harness cannot be forced out of it. Use the
  bench decode window for those rows.