# E053 - first decode-window kernel stats from a real turn, on 4 cards

Artifact: [results/user/llama-cli-traces/82bc067/kernel-stats.log](../results/user/llama-cli-traces/82bc067/kernel-stats.log)
(bench box, `82bc067`, `rocprofv3 --selected-regions --marker-trace --kernel-trace --stats`,
`GGML_PROF_REGIONS=1 GGML_PROF_DECODE=1`, `Q4EXP_POOLED=1 Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1`,
`-sm tensor -fa 1 -lzm on-direct -c 131072 -b 2048 -ub 1024`, large prompt).

The tool's own output file is the pretty table, which is **truncated**: its `%` column sums to 82.98, so
24.3 s of 142.8 s of device time (17%) is in rows it did not print. Everything below shares percentages
against the 118.5 s that is visible, and flags where that matters. Next run: keep the whole `-o`
directory, the CSV stats have every row.

## The window worked, and what it captured

- **No prefill at all**: not one `mul_mat_q` / `mul_mat_id` / batched-MoE row, only `mul_mat_vec_*`. The
  resume/pause gate from `0f641cc1b` is doing its job.
- **1736 decode steps**, derived from the call counts (56 of 59 kernels are exact multiples of 6944 =
  4 x 1736) and cross-checked by the one number that has to be right independently:
  `666624 / (1736 x 4) = 96.0` collectives per step per device, i.e. exactly E037's 2 per layer x 48
  layers. So the unit is steps and the split is per device, not something stranger.
- **rtile is running**: `flash_attn_rtile<256, 4, (ggml_type)1, true>` - D=256, `ncols2` 4, K type f16,
  `use_sparse=true` - 41664 calls = 6 per step per device, which is the 12 QSA layers landing on 2 of the
  4 devices, as the KV-head split implies. The earlier shallow run showed only `flash_attn_tile` because
  with ~36 cells in cache the sparse selection never engaged, not because rtile was unavailable.

Depth (`n_kv`) is not in this artifact, and the run's `llama-cli.log` here is the *previous* shallow run's
(19:38 vs 19:54), so t/s and busy-fraction are not computable from it. What is safe to say: the sparse
path engaged, so depth was above the 2048 budget.

## Where decode device time went (of the visible 118.5 s)

| group | ms | share | calls | note |
|---|---|---|---|---|
| `ncclDevKernel_Generic_4` | 40434 | **34.1%** | 666,624 | 5.82 ms per step per device, 60.7 us per call |
| quantized weight matvecs (`mul_mat_vec_q`, 7 rows) | 48943 | **41.3%** | 2,527,616 | all weight matvecs at n_rows 1 - experts *and* the dense half; the per-type split is below, and the experts-vs-dense share is not yet identified |
| f32 + bf16 matvecs | 7273 | 6.1% | 930,496 | HC, norms-side, indexer projections |
| **whole sparse chain**: `mask_to_sparse_indices`, `flash_attn_rtile`, `combine_results`, `top_k_radix_*` | 3745 | **3.2%** | 1,041,600 | see below |
| micro elementwise (`quantize_q8_1`, `scale_f32`, `k_bin_bcast`, sigmoid, fills) | 5505 | 4.6% | **4,818,368** | 28% of all launches, avg 1.1-1.4 us each |
| norms + hyper-connection | 4990 | 4.2% | 3,027,584 | `dsv4_hc_pre/post`, `rms_norm_f32` |
| recurrent (36 layers) | 834 | 0.7% | 499,968 | `gated_delta_net` + `ssm_conv` |
| copies (`__amd_rocclr_*`) | 1455 | 1.2% | 1,097,152 | |

Per step per device, the visible rows sum to **17.07 ms of busy time**.

## What this changes in the picture

**The attention side is nearly spent.** Selection plus sparse attention plus the gather is 3.2% of decode
device time, and E042/E043 had measured the same chain at 11.1 ms/token/GPU (35% of the qsa arm) *before*
the pool. The pool was supposed to take the gather and pooling out and leave the top-k: that is exactly
what is visible - `top_k_radix_*` (init/histogram/select/gather/reset) survives, and no `copyBufferRect`
storm or pooling-add rows remain, at 6 rtile launches per step per device for 12 QSA layers. Two
consequences worth stating plainly:

- H15's f16 pool storage and H17b's "is the chain replicated 4x" now have a *ceiling* of about 3%, not
  the 35% that motivated them. H17b is still a legitimate correctness question about the cost model, but
  it is no longer a perf priority, and this is the measurement that says so.
- The two things that dominate are depth-independent by construction: the collective (34%) and the expert
  weight reads (41%). That is L3 and H16 - and H16 is parked by the user, so the quantized-matvec group is
  the live one, which is a change from every previous ranking in this directory. **Corrected below: the 41%
  is all weight matvecs, not "experts" - that label was mine and unverified.**

**The launch tail is bigger than it looks.** 4.8 M micro-kernel calls in 1736 steps is 2,776 launches per
step across 4 devices, ~694 per device per step, each 1.1-1.4 us. They total only 4.6% of device time, so
this is not a "kernels are slow" finding - it is a standing measurement of how much per-step work is
`op_count`-shaped, which is the number that would make graph-level fusion (N1, H2) worth anything. H14
closed "dispatch-bound?" as no at 79% device time; this refines it: the device time is real, and it is in
~700 sub-2 us launches per card per step.

**One caution before comparing runs.** The shallow run in this same directory (`llama-cli.log`, 9-token
prompt) reported `phase:decode` min 7.03 ms untraced vs 20.78 ms min under the same trace flags, so
attaching inflates a step by roughly 14 ms at that size. Absolute per-step ms from any of these traces is
therefore upper-bounded by the tool's cost, and shares across two traced runs are comparable while their
totals are not.

## Are we dequantizing too many experts? No - and that reframes the 41%

Asked directly, because 41% of decode device time in weight matvecs is either "too many bytes" or "bytes
moving too slowly", and only one of those is an mmvq problem.

**Too many experts is ruled out by launch counts, with two orders of magnitude of margin.** Reading all 512
experts in all 48 layers would be ~24,576 launches per card per step. The whole table has 2,450 launches
per step per card, of which 519 are quantized matvecs at all. So routing works: we touch the 10 active
experts, and the byte count follows from that -

| per token, whole box | bytes |
|---|---|
| `ffn_down` 48 x 10 x (640x2560) at Q5_0 | 0.59 GB (0.44 GB if it were Q4_K - see the new `quant-block-size-fallback` memory) |
| `gate`+`up` 48 x 10 x 2 x (2560x640) at Q4_K | 0.88 GB |
| GDN in/out for 36 layers, attention for 12, shared expert, `lm_head` (248,320 vocab) | ~1.4 GB |
| **total, per card = 1/4** | **~0.8 GB** |

**The gap is inside the kernels.** 0.8 GB per card per step at the R9700's ~640 GB/s is ~1.3 ms. The
quantized matvecs take **7.05 ms per card per step**, i.e. ~115 GB/s achieved, **roughly a fifth of peak**,
and the f32/bf16 matvecs add 1.0 ms more. Per-call cost is 5.5-29 us for tensors whose slices are single-digit
megabytes, so this is not launch latency either - each kernel is moving its own bytes slowly.

So the framing changes: not "MoE traffic is 41% of decode, go make routing smarter", but **"n_rows == 1
weight reads run at ~20% of achievable bandwidth"**, which points at mmvq's RDNA4 configuration
(H6's `mmvq.cu:417-492` nwarps whitelist and the `prefer_f32_output` at `ggml-cuda.cu:1512`) and at the
519 Q8_1 activation-quantize launches per card per step (0.58 ms, i.e. ~8% of the matvec time is spent
re-quantizing the activation once per matvec rather than once per width).

Caveats worth keeping: the byte table is derived from the architecture, not measured, and it assumes
`ffn_down` is the Q5_0 row - the loader's type census settles that in one line and is still outstanding.
The wall time per step for *this* run is also not in the artifact (see above), so the busy fraction is not
computable here; on the shallow run it was ~34%.

## Still needed from the box

- the run's actual `n_kv` / prompt token count, and its `[prof] phase:decode` line, for busy fraction
- the full `--stats` output (whole `-o` dir), to identify the 24 s the table hides
- `GGML_FATTN_DEBUG=1` once, to record `kernel=rdna_rtile D=256 n_q=1 n_kv=.. gqa=.. n_kv_max=2048` as
  the positive identification of the shape the sparse decode actually runs at on 4 cards