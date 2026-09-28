# E065 - dev-box phase inventory: what the pp and tg walls are made of (1x gfx1201)

- date: 2026-09-28
- machine: dev-rx9070-16g (1x gfx1201), ROCm 10.0.0, `-ngl 99`
- tier: T1
- status: running
- parent: none (baseline). Reads against E063's bench anchor, E059/E061's kernel work, the
  `GGML_PROF_WINDOW` addition `fc61603f0`, and the removal of llama-bench's implicit window `cc6be7d35`.
- raw: [results/E065-dev-box-phase-inventory/](../results/E065-dev-box-phase-inventory/)

## question and what decides it

With the current defaults on one card: what is a prefill's and a decode step's *wall* made of - host
phases or GPU math - and which kernels own the GPU part? This is an inventory, not an A/B, so no effect
size is claimed and the deciding metric is a *composition*:

- the `[prof]` region table's `pp` / `tg` columns, as a share of the phase wall that the same table
  reports (`phase:prefill` / `phase:decode` totals)
- per-kernel device time from `rocprofv3 --selected-regions`, grouped into the families we track
  (mmq/mul_mat_q on the pp side; mmvq/mmvf and the hc/qsa rows on the tg side)

Prediction, written before running: pp is GPU-bound with a single-digit host share, `graph:compute` at
about the whole wall; tg shows a visible non-GPU share (`phase:sync`, the lazy io regions) with its GPU
part led by the mmvf/mmvq matvecs and the `hc_*` and QSA/indexer rows behind. Falsified if pp's host
share tops ~10% or tg's non-GPU share comes in under ~5%.

Out of scope: no A/B, so no interleaving or effect-size discipline applies; the traced arms are used for
kernel composition only and never for a wall number (E061's rule). The qsa pool mode line is not checked
(needs `-v`), so this round says nothing about pooling.

## arms

Harness is `llama-cli`, so batching is the server's: `-b 2048 -ub 512` are the defaults, i.e. an
8192-token prompt arrives as 16 ubatches of 512.

| arm | env | reps |
| --- | --- | --- |
| pp-untraced | `GGML_PROF_REGIONS=1` | 3 |
| pp-traced | `+ GGML_PROF_WINDOW=pp`, under `rocprofv3 --selected-regions --marker-trace --kernel-trace --stats` | 2 |
| tg-untraced | `GGML_PROF_REGIONS=1` | 3 |
| tg-traced | `+ GGML_PROF_WINDOW=tg`, under the same tracer | 2 |

All arms: `LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib` (ROCm 10 has no loader entry), one GPU so no
comm path is exercised. Correctness gate: no code is changed and no perf delta is claimed by this round,
so none is required; the golden PPL on this same toolchain is `263100.7437`.

## commands

[`results/E065-dev-box-phase-inventory/commands.sh`](../results/E065-dev-box-phase-inventory/commands.sh)

## harness correction, and a retraction

The pre-registered harness passed the whole sparse corpus (32.68k tokens) with `-c 8192`, which
**aborts**: `Error: request (32673 tokens) exceeds the available context size (8192 tokens), try
increasing it`. llama-cli errors instead of truncating, and after the error the `[prof]` table holds
only the tokenizer warmup - 2 calls / 4 tokens.

That 4-token reading is what an earlier version of this section blamed on `-f` ("`prompt_file` is never
read"). **That was wrong and is retracted**: with `-c 65536` the identical `-f <corpus>` invocation
measures `tok:prefill 21 calls 32.68k total`, so `-f` delivers the corpus exactly as the user said. The
grep finding (`common_params::prompt_file` has no reader outside imatrix/cvector-generator) describes a
different field and says nothing about the cli's own file handling. The claim that the user's older
`results/user/llama-cli-traces/*` runs profiled decode at depth 0 is retracted with it - those runs used
`-c 131072` and were fine.

What survives: `-c` must exceed the prompt, and the depth comparison needs a prefix that fits. The
shallow arms below therefore run a 45k-byte prefix of the corpus (10.90k tokens) in `-c 16384`.

## results

### prefill, untraced (llama-cli, `-b 2048 -ub 1024`, `-fa 1 -ngl 99`)

Six measurements per depth (3 pp-only, 3 tg-with-pp; all medians are over those, rep 1 of the deep set
being a cold page cache at +5%):

| depth | wall ms | `graph:compute` | `sched:realloc_size` | `graph:set_inputs` | other |
| --- | --- | --- | --- | --- | --- |
| 32.68k (21 calls, 1.56k/call) | 5481 | 3408 (62%) | 1689 (31%) | 326 (6%) | 58 |
| 10.90k (10 calls, 1.09k/call) | 3642 | 3079 (85%) | 490 (13%) | 40 (1%) | 33 |

The realloc is **per prefill call and stable to +-0.5%**: 80.4-81.0 ms/call at 32.68k, 48.9-49.0 ms/call
at 10.90k (one 62.4 ms outlier in the shallow set, whose wall was also high). It is `graph:alloc` almost
exactly (`sched:realloc_size` is 1685 of 1689 ms), so the scheduler is re-allocating its graph buffers on
every batch rather than reusing the previous one; `graph:build` is 5 ms and `graph:reuse` 0.2 ms, i.e.
the reuse test itself is free. Per token this is 52 us at the deep depth and 45 us at the shallow one,
while the *rest* of the wall per token is 2.4x higher at the shallow depth - the cli splits the two
prompts into different call widths (1.56k vs 1.09k tokens/call), so per-token rates between the two dep
depths are not comparable. Per call they are.

### decode, untraced

| depth | steps | cli | `phase:sync` (tg) | `phase:decode` | `graph:compute` (tg) | `set_inputs` (tg) | `graph:alloc` (tg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32.68k | 94 | 239.4 t/s = 4.18 ms/step | 3.17 ms/step | 0.72 ms/step | 0.55 ms/step | 0.15 ms/step | 0.00 ms/step |
| 10.90k | 255 | - | - | 0.62 ms/step | - | - | - |

A step is ~3.2 ms of GPU drain (`phase:sync` carries it; `phase:decode` only enqueues) plus ~0.6-0.7 ms
of host work plus sampling, so the non-GPU share is roughly a quarter of a step - the prediction's
"visible" bar was 5%. The dummy stops early at the deep depth (EOS after 94 of 256 requested) and runs
all 255 at the shallow one.

### traced: kernel composition (never a wall)

Two reps, dispatch-identical (14471 kernels pp, 34780 tg; busy 2035.5 vs 2036.9 ms pp, 246.0 vs
245.9 ms tg).

**prefill, 32.68k tokens** - 2036 ms of device time, top kernels:

| kernel | calls | ms | us/call |
| --- | --- | --- | --- |
| `mul_mat_q` | 495 | 438.4 | 886 |
| Tensile `Cijk_..._HSS_...` (GEMM) | 1056 | 282.4 | 267 |
| `gated_delta_net_cuda<128,false,false>` (SSM) | 108 | 244.2 | 2261 |
| Tensile `Cijk_..._S_B_...` (F32 GEMM) | 331 | 175.2 | 529 |
| `flash_attn_ext_f16<256,256,1,16>` | 29 | 140.9 | 4859 |
| `k_bin_bcast` | 1338 | 83.0 | 62 |
| `cpy_scalar` | 180 | 68.6 | 381 |
| `dsv4_hc_post_f32<false>` | 222 | 61.0 | 275 |

GEMMs together (`mul_mat_q` + both Tensile families) are 896 ms = 44% of device time; the SSM gated
delta net 244 ms = 12%; flash attention 141 ms = 7% for only 29 calls, i.e. ~1.4 of the 4 layers take
the attention path. The arch's own ops (`gated_delta_net` + `dsv4_hc_post` + `mm_ids_helper` 297 calls /
59.7 ms) come to ~18%, elementwise/norm/copy the remaining ~20%.

**decode** - 246 ms of device time for 94 steps = 2.62 ms/step, top kernels:

| kernel | calls | ms | us/call |
| --- | --- | --- | --- |
| `mul_mat_vec_q` | 4512 | 177.7 | 39.4 |
| `mul_mat_vec_f<__hip_bfloat16,...>` | 1128 | 6.5 | 5.8 |
| `mul_mat_vec_f<float,...>` | 1316 (1410 in rep 2) | 5.9 | 4.4 |
| `quantize_q8_1` | 4512 | 5.8 | 1.3 |
| `k_bin_bcast` | 3384 | 5.4 | 1.6 |
| `__amd_rocclr_copyBuffer` | 2162 | 4.3 | 2.0 |
| `scale_f32` | 3196 | 3.3 | 1.0 |

The matvec family is 190 ms = 77% of decode device time. `quantize_q8_1` appears **once per matvec**
(4512/4512) - H26's mechanism, at 1.3 us a call: only 2.4% of device time here, but **13% of all 34780
dispatches**, which is where its leverage sits.

### traced: occupancy and launch structure

| arm | kernels | busy ms | span ms | idle | busy share |
| --- | --- | --- | --- | --- | --- |
| pp 32.68k | 14471 | 2036 | 6197 / 6288 | 4162 / 4253 | 33% |
| tg (94 steps) | 34780 | 246 | 500.0 / 500.8 | 254 / 255 | 49% |

So the trace agrees with the region table: on prefill the GPU holds work for a third of the wall. With a
1 ms gap threshold the pp timeline splits into 94-95 clusters that are *tiny* - median 13 kernels, 0.18 ms
busy, 1.57 ms span, 1.33 ms gap - i.e. short kernel bursts separated by the host's per-batch stalls.

The decode arms cluster into 2 (one >1 ms hole mid-phase) with a median of 17390 kernels / 123 ms busy /
249 ms span, so **within a step the 370 dispatches/step are essentially back-to-back** (sub-ms gaps) and
the large idle sits *between* steps, where the sampler runs. Per step: 0.72 ms host enqueue, then a
3.17 ms drain in which 2.62 ms of kernels execute and ~0.55 ms is launch gap, plus ~0.3 ms of sampling -
which sums to the measured 4.18 ms.

Tracer perturbation, from the cli's own line: prefill 5565.7 / 5647.5 t/s traced against 5976.8 / 5982.7
untraced (**-6%**), decode 185.9 t/s against 239.4 (**-22%**). Decode is the tracer-sensitive phase, so
its traced gap total (2.7 ms/step, from a 5.33 ms spanned step) overstates the real one - untraced the
kernels are the same 2.62 ms in a 4.18 ms step, i.e. ~1.6 ms of gap.



### hsa / memory trace (one prefill arm, all domains on)

`--hsa-trace --memory-copy-trace --memory-allocation-trace` (plus the usual kernel/marker flags and the
same window) on the pp arm. The flag set works on rocprofv3 1.3.5; the run takes 9.9 s and slows the arm to
5384.9 t/s, so it is a composition instrument only.

| fact | number |
| --- | --- |
| `hsa_signal_wait_scacquire` | 1035 calls, **1847 ms** (75% of HSA API time) |
| `hsa_executable_freeze` + `load_agent_code_object` | 40 + 40 calls, 431.5 + 156.6 ms = **588 ms** |
| H2D copies | 169 / **82.4 ms** (487 us each) |
| D2H copies | 24 / 1.3 ms |
| allocation trace | 13 allocs 1.8 ms, 24 frees 33.7 ms |

1. **The staging is CPU-bound.** `graph:set_inputs` is 326 ms of host region time and the transfers inside
   it are 82 ms of H2D, so ~244 ms is the host gathering input. The HSA copy *enqueue* for all 193 copies
   is 0.6 ms.
2. **The "realloc" is a forced device sync, not an allocation.** 1645 of the region's 1689 ms is
   `hsa_signal_wait_scacquire` (427 waits overlap those regions). The source says why: the branch is
   `if (backend_ids_changed || !ggml_gallocr_alloc_graph(...))` and it calls `ggml_backend_synchronize()`
   for every backend before `ggml_gallocr_reserve_n`, because the buffers may move. The allocator itself
   costs 35 ms.
3. **Why it trips every batch.** `GGML_SCHED_DEBUG_REALLOC=1` aborts at `ggml-backend.cpp:1627`:
   `unexpected graph reallocation (graph size = 702, nodes = 702, leafs = 141)`. The region is always
   `realloc_size` and never `realloc_buft`, so the buffer-type assignment is stable and so is the node
   count; the only remaining input to `ggml_gallocr_needs_realloc` (`ggml-alloc.c:1053`) is a **tensor
   byte-size change** - some tensor's `nbytes` differs between batches while the topology does not. So
   this is a *sizes* trip, not a growing topology. Follow-up: log which tensor changes.
4. **588 ms of the run is code-object loading** (40 kernel first-launches). A long-lived server pays it
   once per process; these single-prompt runs pay it inside the wall, so discount it when comparing to
   server behaviour.

## reads against

- The pre-registered prediction is **falsified on prefill** (predicted GPU-bound with a single-digit host
  share; measured 31% per-call scheduler realloc and the GPU idle two thirds of the wall) and
  **satisfied on decode** (visible non-GPU share, matvec-led GPU part).
- This is a T1 number on a **4-layer all-F32 dummy** (11.88 GiB): the *kernel mix* is not the quantized
  real model's, so only the host-side structure transfers - the per-call realloc churn, the staging, the
  sync split, and the host share of a step. It says nothing about the real model's GPU composition.
- H19's territory, corrected by the HSA data above: the realloc branch fires per batch, but 97% of its
  region is the forced drain of the previous batch's kernels - time that has to pass anyway. What the
  branch actually costs is the *overlap* it forbids: the staging (326 ms), build (5 ms) and checkpoint
  (109 ms) that could have run during that drain, i.e. up to ~0.4-0.5 s of the 5.5 s wall, **not** the
  1.7 s an earlier reading of the region suggested. Its root cause is a per-batch tensor-size change, which
  is what a fix has to attack.
  while the call count grows to 21). If a later arm removes it, the pp wall has ~1.7 s to give back at
  32.68k on this box, and the check is the same host-side table, no tracer needed.
- H26's mechanism reproduces on the dev box (1 q8_1 quantize per matvec) at 4 layers' scale.
- Not measured here: the pool mode line (needs `-v`), copy *volumes* (the copy trace has no size column
  with these flags) and no A/B was run, so no effect size is claimed.
- Trace side: the decode step is not launch-starved - 370 dispatches/step run back-to-back for 2.62 ms of
  the 4.18 ms step - so the host cost is real work (enqueue + sampler), not a launch wall. The prefill is
  the opposite: bursts of 13 kernels averaging 0.18 ms with the GPU idle two thirds of the wall.

