# Plan: what prefill actually costs, and where the next win is

Scope: qwen4exp prefill on the bench box (4x R9700, `-sm tensor`, real Q4_K_M) with the dev box as an
overlay. Written 2026-09-28 at `bba53a6c3`. This is a map, not an experiment: no effect size is claimed
except where a record is cited. Threads are local ids `P1..P8`; each becomes an `E<nnn>` when it has a
falsifiable hypothesis and a deciding metric, per `../PROTOCOL.md`.

Motivation: E074 fixed the realloc ratchet, E077 moved the n-gram fetch off the critical path. What is
left is `graph:compute`, the largest single region on the box, and it had never been decomposed. This
file says what the existing traces do and do not cover, what a ubatch is made of, and which candidates
are worth an experiment.

## 1. The one measurement this plan turns on

No trace of the real model's prefill exists, with or without a window. The five surviving decode CSV
families (`results/user/h23-runs-*, results/user/llama-bench/d133df7d4`) contain zero `mul_mat_q`,
`mm_ids_helper` and Tensile rows by their own markers, and the only real-model traces that do contain
prefill rows are the E037/E038 `qsa-kernel-traces*` logs, which are mixed with ~385 decode tokens, come
from `b827606c8`, and whose arms differ in sparse FA and indexer use (whether the pool was on is not
recorded in those files). So run this once, on the box, when it is free:

```sh
GGML_PROF_REGIONS=1 GGML_PROF_WINDOW=pp rocprofv3 --kernel-trace --marker-trace --stats -o /tmp/pp -- \
  llama-bench -m <real model> -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU \
    -d 16384 -p 4096 -n 0 -r 5 -b 2048 -ub 1024 -v
```

Read three things, in this order:

1. `graphs reused` in the perf printout, and whether `graph:reuse` appears in the region table.
2. Bursts: sort dispatches by start, split on gaps > 0.5 ms, and report kernels, busy ms and span per
   burst. The dev-box run below shows the method (4.6-4.8k kernels per ubatch at 96.5-97.2% busy).
3. The family table of the current build, next to the A5 table in section 3.

Decision rule: if the box's bursts are ~95% busy and longer than `graph:compute` per ubatch (139 ms),
the device is the wall and P3-P5 lead. If they are short and gappy, the host path is the wall and P1,
P2, P6 and P7 lead. `phase:sync` (97.7 ms/ub) is the drain tail and is neither: it is the device time
that is *not* overlapped. Anchor on the shape question, measured on the dev box with the 48l dummy and
`-v`: `graphs reused = 0` for `-p 4096 -n 0` and `graphs reused = 126` for `-p 512 -n 128 -d 4096`, i.e.
prefill rebuilds every ubatch while decode reuses all but two steps.

## 2. Budget of a prefill ubatch today (box, real model, 1024 tokens)

All per call (2048 tokens) as measured, halved for the per-ubatch column. Sources: E074 (`ede6d511c`,
pre-E077) and `results/user/e79-02dd5cea2/run-prof.log` (`02dd5cea2`, post-E077).

| region | per call | per ubatch | note |
| --- | --- | --- | --- |
| `graph:compute` | 278.65 ms | 139.3 | wraps `ggml_backend_sched_graph_compute_async`, includes the two rows below |
| `meta:subgraph` | 26.6 (0.2754 x 96.5) | 13.3 | split dispatch, inside compute |
| `meta:allreduce` | 12.3 (95.5 calls) | 6.2 | host side of the collectives, inside compute |
| `phase:sync` | 195.31 | 97.7 | device drain, not overlapped |
| `graph:set_inputs` | 53.50 post (89.20 pre) | 26.8 | includes gather 43.54 post (79.18 pre), staging 0.9, h2d 0.7 |
| `graph:alloc` | 20.47 | 10.2 | only when the graph is not reused, i.e. always |
| `graph:build` | 1.01 | 0.5 | ditto |
| `mctx:apply` | 0.28 | 0.1 | |
| graph | 6825 nodes, 1372 leafs | - | pooled reserve, E049/E056 |

Not in the table because it is not measured: the device's own busy time per ubatch. The old capture
(`results/user/amd-smi-prefill.log`) shows 34% GFX utilisation during prefill with 93/210 W per card,
but that predates E077 and the user reports it higher now, so treat it as unset.

`graph:alloc` is three things, and on the box it is the meta buffer's per-tensor init. Measured with the
`sched:*` regions and the `meta:init_*_ns` counters (`results/user/llama-bench/bee859d04/run-prof.log`,
56 ubatches of 1024 tokens): `graph:alloc` 24.10 ms/call, of which `sched:init_nodes` 22.25 and
`sched:split` 0.88, over ~6800 meta inits per ubatch. Phases per init: prep 1.65 us (container lookup plus
split state), tail 435 ns, create 507 ns (per sub-buffer create, copies, buffer init), srcs 349 ns. The dev
box (1 sub-buffer) shows 1.94 / 431 / 170 / 185 ns, so only create and srcs scale with the device count.
That is 20.0 ms/ubatch, 4.1% of a 489 ms ubatch, and P9 is where it is chased.

## 3. Real-model device composition, best available (A5: sparse FA + rtile, d131072)

`results/user/qsa-kernel-traces-postfix/stats-qsa-rtile-131k.log`, build `b827606c8`, a 131072-token
depth fill plus 385 decode tokens, so it is ~prefill. Shares, not ms: `--stats` may replay kernels.

| family | calls | ms | share |
| --- | --- | --- | --- |
| NCCL collectives | 98,436 | 42,601 | 16.3% |
| `mul_mat_q` (mmq) | 135,168 | 38,085 | 14.6% |
| Tensile GEMMs | 303,104 | 34,763 | 13.3% |
| unary (sigmoid/silu/softplus) | 760,276 | 22,708 | 8.7% |
| HC (`dsv4_hc_pre` + `_post`) | 196,608 | 20,034 | 7.7% |
| `k_bin_bcast` | 282,116 | 17,779 | 6.8% |
| `mm_ids_helper` | 72,192 | 15,230 | 5.8% |
| `rms_norm` | 250,320 | 10,265 | 3.9% |
| `gated_delta_net` | 37,008 | 9,090 | 3.5% |
| `k_get_rows` | 103,568 | 7,742 | 3.0% |
| dense FA | 6,288 | 7,077 | 2.7% |
| `top_k_radix` | 133,056 | 5,763 | 2.2% |
| `quantize_*` | 579,868 | 5,416 | 2.1% |
| moe weighted reduction | 45,232 | 4,677 | 1.8% |
| mmvq / mmvf / copyBuffer / rope / fill / rest | - | ~10,000 | ~4% |

The no-qsa twin of the same run: NCCL 28.8%, **dense FA 17.0%** against 2.7% here, so the sparse/rtile
work already bought about 14 points of device time at depth. These ranks are for a 131k context; at
shallow depth the attention share falls and the GEMM, HC and dispatch shares rise.

Dev-box overlay, current build, 48-layer dummy (6873 nodes / 1372 leafs, structurally the real model,
but F32 and small): 76,554 dispatches over 16 ubatches = **4,785 launches per ubatch**, each ubatch one
burst of 4,611-4,813 kernels at **96.5-97.2% busy**, spans 539-640 ms, and all 32% of the run's idle in
110 gaps >= 5 ms, i.e. between ubatches. `graphs reused = 0`. Device split: Tensile 21.2%, GDN 19.0%,
mmq 16.5%, HC 7.1%, FA 6.0%. Launch leaders are dummy artifacts (`convert_unary` f32->f16 418/ub,
`dequantize_block_*` 406/ub, `scale_f32` 381/ub) plus real ones (`unary_op` 277/ub, `k_bin_bcast`
174/ub, `mm_ids_helper` 141/ub, `quantize_mmq_q8_1` 118/ub, `rms_norm` 99/ub, `hc_pre` 96/ub).

## 4. Launch census, real model (source-derived, +-20%)

~3,200 launches per 1024-token ubatch, over 6825 graph nodes; about half the nodes are free views.

| family | launches/ubatch | granularity |
| --- | --- | --- |
| HC (2 mixes + 2 combines) | **~1,056** | 22 per layer, 6 of them GEMMs |
| linear-attention block (36 layers) | ~760 | 21 per layer |
| MoE + shared expert | ~672 | 14 per layer |
| QSA chain + attention (12 layers) | ~400 | ~33 per layer, 11 of them TOP_K radix |
| PLE (1 layer) | ~45 | |
| prelude / head | ~10 | |
| **total** | **~3,200** | matmuls ~890: MMQ ~330, hipBLAS/Tensile ~560 |

## 5. Structural facts that decide the candidates

### 5.1 HC is the largest launch consumer and runs replicated four times

`hc_*` and `ple_*` are unmatched in the split table, so they are MIRRORED: no split, no allreduce, full
work on every card (`src/llama-model.cpp:541,568,579-583,605-606`; PARTIAL is only created when both
operands are split on the contraction dim). HC is ~27 GFLOP/layer/card against the MoE's ~17 GFLOP
k-split. Its tail is unfused: `SCALE,SIGMOID,SCALE` is 288 launches and `SCALE,SILU` 192 per ubatch.

### 5.2 The allreduce count is a hard gate

`attn_output`/`ssm_out` and the merged MoE+shared down branch produce PARTIAL state, 2 per layer = 96
per ubatch, matching `meta:allreduce` 95.5 per call. Nothing here is an estimate.

### 5.3 Attention has a cliff at ~4104 KV cells

The rtile hook needs `Q->ne[1] <= 16`, so it is rejected at prefill widths; sparse FA engages only when
`K->ne[1] >= max(4096, 2*top_k)`, i.e. 4104 cells. Shallow prefill therefore runs dense
`flash_attn_ext_f16<256,256,16,4>`, deep prefill runs sparse `<256,256,1,16,true>` with a compaction
pass. That is exactly A5 against A6.

### 5.4 At prefill width every quant type takes MMQ

The RDNA4 cutoffs in `mmq.cu:380-407` are all `ne11 <= 512` or less, and Q8_0/IQ*/MXFP4 fall through to
`return true`; `MUL_MAT_ID` is forced onto MMQ by the `n_experts > 0` clause. So H10's question is the
*tile* choice for MoE shapes, not path selection: `ncols_opt = ceil(1024*10/512) = 21`.

### 5.5 The prefill quantize share is larger than H26 assumed

H26 priced prefill's redundant `q8_1` at ~0.9% against a 461 ms ubatch. A5 shows `quantize_*` at 2.1%
over 579,868 calls. Same family, different size, and it is a launch item more than a device item.

### 5.6 Round-trip check: `ffn_down_exps` is not uniformly Q5_0

`results/user/gguf-dump.log` has 24 tensors at Q8_0 and 24 at Q5_0 for `ffn_down_exps` (gate/up are all
Q4_K). The 640/256 fallback is real, but this file is a per-layer imatrix mix, so the "N = n_layer x
n_expert Q5_0 tensors" signature must not be used as a check here. The memory file was corrected.

## 6. Candidate threads

### P1 - stop doing HC four times, and fuse its tail

- **Question:** can the HC GEMMs take the split path (partial + one reduce) instead of being mirrored,
  and can `SCALE,SIGMOID,SCALE` / `SCALE,SILU` become one node each?
- **Prize:** 33% of the launch census and 7.7% of device time, of which three quarters is redundant
  across cards if the GEMMs are compute-bound; plus 480 launches per ubatch of pure elementwise.
- **Next step:** a counter of HC GEMM time and work per card, then a split-table change for `hc_*`;
  check the accuracy gate (a partial sum in a mirrored op changes the order of operations).
- **Deciding metric:** pp t/s on the box, with device time per card from the trace.
- **Risk:** `hc_*` is F32-only by gate (N5) and feeds the residual stream, so a split introduces one
  reduce per mixture and could lose more in collectives than it wins.

### P2 - freeze the QSA window so prefill can reuse its graph (and then replay it)

P2a was attempted on 2026-09-28 and retracted: windowing the QSA block domain is safe but does not produce a
reuse hit, because the remaining blockers are cell-domain inputs. The detail and the numbers are below;
P2b's precondition is therefore missing for now.

#### P2a - the reuse half: attempted 2026-09-28, retracted, the 22 ms/ub is not reachable this way

- **Prerequisite, measured.** The prize exists: `results/user/e79-02dd5cea2/run-prof.log` (56 ubatches of
  1024 tokens) has `graph:alloc` 56 x 20.95 = 1173 ms and `graph:build` 56 x 1.04 = 58 ms, i.e. **22 ms per
  ubatch**, which at pp4096 d16384 (2096 t/s, 489 ms/ubatch) is **4.5% of prefill**. A reuse hit skips both:
  `llama_context` does not call `graph:alloc`/`graph:build`, and `ggml_backend_sched_graph_compute_async`
  finds `sched->is_alloc` still set, so it runs `compute_splits` with the previous split list and buffer
  assignment (`ggml-backend.cpp:2025-2065`). The region table's `ms/call` column is per ubatch, not per
  2048-token call.
- **Attempt.** An env-gated `Q4EXP_QSA_WINDOW` (tokens, default off) widened the pooled block width:
  `blk_cells`, `bias` and the pooled score/top-k chain were sized from a rounded-up block count (the bias
  already writes `-INFINITY` past the live range, `llama-memory-hybrid-idx.cpp:939-946`), the score got a
  clamp so a garbage pool row cannot meet that -inf as a nan, `can_reuse` used the same width, and the window
  was allowed only for prefill (`n_tokens > 1`) and only once the selection is saturated (otherwise top-k would
  pick a padded block whose cell entry is 0). Patch kept at
  [results/P2a-qsa-window-attempt/qsa-window.patch](../results/P2a-qsa-window-attempt/qsa-window.patch),
  which applies to the tree at `02dd5cea2`, is 301 lines over 5 files, and also carries the two
  `Q4EXP_REUSE_DEBUG` prints that found the blockers below; all source files reverted afterwards.
- **It was safe but useless.** Correctness held: on the dev box with the 4l model and the sparse corpus at
  `-c 8192`, per-chunk and final PPL are bit-identical with the window on and off
  (270867.4519 / 270355.0602 / 266980.7507). No re-reservation either, once the clamp node was gated on the
  env instead of on the width. But `graphs reused` stayed 0 in every arm.
- **Two blockers, found with an instrumented `can_reuse`.**
  1. With llama-bench's chunking (`-b 2048 -p 4096`) the reuse check is never entered at all: the two
     `gf_res_prev` slots alternate, because the output flag alternates ([0,1,0,1] - logits are requested on the
     last token of each chunk). `gf_res_prev_active == res` is then false by construction. With `-b 8192` (one
     call, pattern [0,0,0,1]) the check is entered.
  2. Entered, it fails on 2 of 6 inputs, identically with the window on and off: the n_kv-sized cell-domain
     inputs (the KQ mask via `can_reuse_kq_mask`, and the QSA/attention input). `allow_reuse` itself returns 1.
- **Why windowing those is not the answer either.** Padding the mask and the K/V views makes the dense
  attention process masked cells: a W-token window adds about W/2 masked cells per ubatch, so FA's work grows
  with the window until the sparse path takes over (above ~4104 cells it gathers a fixed top-k set and stops
  tracking n_kv). That is device work bought to save host time, and how the two trade depends on the host
  pass-through, which nobody has measured. (An earlier draft of this section said +10% device time, which
  conflated the device share with the wall; corrected 2026-09-28.) So prefill reuse needs the cell window, and
  the cell window is only cheap at depth, where the host bill is smallest.
- **Constraint learned for any future attempt.** A node that appears only at runtime breaks the measured
  graph: with the clamp gated on `n_blocks_g > n_blocks`, the runtime topology differed from the reserved one
  and every prefill ubatch re-reserved (`sched:realloc_buft` at ~518 ms/call). Any node added by a window must
  be present in the graph the reservation measures (gate on the env, not on the width).
- **What is left of P2:** P2b below, whose precondition (a reuse hit) is now known to be missing, and the
  cheap question underneath it - why the split plus gallocr pass costs 22 ms/ubatch at all, and whether the
  node count can come down (P1, P5) so that the price falls without reuse.
- **Cost, and why the window size is the whole question.** The score, the bias fill and TOP_K all scale with
  the padded block count. The bias is already 16.8 MB per QSA layer per ubatch at d16384, x12 layers, so a
  window doubles that before any extra scoring. Padded pool rows were never written, so a `ggml_clamp` on the
  scores before the bias add (`fminf(fmaxf(x,min),max)`, NaN-safe) is needed or a garbage +inf meets -inf and
  turns into NaN inside top-k.

#### P2b - the capture half (conditional)

- **Mechanism.** `ggml_backend_cuda_graph_compute` only captures once two consecutive calls have identical
  node properties - the whole `ggml_tensor` of every node plus each source's data pointer, `ne` and `nb`
  (`ggml-cuda.cu:2673-2714`) - and a re-split assigns fresh split uids (`ggml-backend.cpp:1604`), so the
  uid fast path needs reuse as well. Prefill satisfies neither today: properties differ every ubatch, the
  warmup never completes, and the host issues all ~3,200 launches itself.
- **Prize if it lands:** the launch API cost inside `graph:compute`, which needs the box trace to price; the
  device work is unchanged either way.
- **Caveat measured on the dev box (48l dummy, interleaved, `-r 5`, tg128 @ d4096):** capture on
  42.01 +- 0.13 and 41.93 +- 0.10 against capture off 42.52 +- 0.05 and 42.40 +- 0.04, i.e. **capture is
  1.2% slower there**, the opposite sign to E008's box measurement ("graphs off costs ~7%", older stack).
  Same-config pp4096 was inert (`GGML_CUDA_DISABLE_GRAPHS=1` -0.66%, inside its own band). So capture's value
  is stack- and shape-dependent and must be re-measured on the box before P2b is treated as a prize. One
  `GGML_CUDA_DISABLE_GRAPHS=1` tg arm would settle it, and if the dev-box sign reproduces it is a free 1-2%
  at decode with no code change.

### P3 - MoE tile and GEMM choice (H10, now priced)

- **Question:** are the `mul_mat_id` tiles right for the expert shapes at `-ub 1024`?
- **Prize:** mmq + Tensile + `mm_ids_helper` is 33.7% of device time. Your own partial retune showed
  pp512 -11.2% at 40k on the dummy.
- **Next step:** land or revive the RDNA3/4 mmq retune for MoE shapes; nothing new to measure first.
- **Deciding metric:** pp at d4096 and d40960, interleaved.

### P4 - prefill-width collectives

- **Question:** is NCCL the right collective at 10.5 MB per message, and can it overlap the next layer?
- **Prize:** 48 collectives per ubatch at 432.8 us each = 20.8 ms/ubatch if fully exposed, 16.3% of
  device time in A5. The in-tree one-shot path is opt-in and its 256 KiB cutoff is a guess.
- **Next step:** one arm with the internal one-shot path at prefill width, and a check of the effective
  bandwidth (A5's 10.5 MB in 432.8 us is ~24 GB/s, ring-like).
- **Deciding metric:** pp t/s, plus `meta:allreduce` and the collective kernel time per ubatch.

### P5 - the fusion tail and the elementwise family

- **Question:** which of the remaining chains deserve a pattern?
- **Prize, by count:** HC combine 288, HC `SCALE,SILU` 192, GDN l2 `RMS_NORM+SCALE` 144 (the existing
  pattern wants MUL, not SCALE), QSA pooling `CONT x4 + ADD x3 + SCALE` 96, TOP_K radix 132, N1's
  IMROPE `RMS_NORM+MUL+ROPE` gap on 12 layers. Device side: unary 8.7%, `k_bin_bcast` 6.8%,
  `rms_norm` 3.9%. The QSA mask rebuild is `FILL` + `SET_ROWS` + `ADD` over `[n_kv, 1024]` f16, about
  268 MB per QSA layer per ubatch at 131k, so it is a bandwidth item, not a launch item.
- **Next step:** start with the patterns that need no new kernel math (`SCALE,SILU`; `RMS_NORM,SCALE`;
  the GDN and QSA chains), and leave IMROPE last because the fused kernel has to implement the
  interleaved rope.

### P6 - split dispatch: `meta:subgraph` 13.3 ms/ubatch

- **Question:** what does a subgraph dispatch cost, and are the cross-backend input copies at prefill
  width (a 1024x2560 f32 activation is 10.5 MB) going through P2P?
- **Prize:** 13.3 ms/ubatch of host time plus whatever the copies cost, both inside `graph:compute`.
- **Next step:** count copies and their volume in the box trace (`__amd_rocclr_copyBuffer`, SDMA rows)
  and check whether `GGML_CUDA_P2P` is in play. Prior art: `results/user/p2p-improvements/`,
  `p2p-legacy-improvement-plan.md`.

### P7 - `set_inputs` leftovers

- **Question:** what is left in the 26.8 ms/ubatch?
- **Prize:** gather is 43.5 ms/call post-E077 (a covered window costs ~8 ms against ~79 for an
  uncovered one), so the remaining half is the staging fill and the QSA host mapping. E056 measured
  90.8 against 125.9 ms/call at 131k for the pooled path, and E028 prices the leftover QSA mapping at
  ~3.4 ms/token at 131k.
- **Next step:** the staging-fill extension offered in E077 (thread does reads and `to_float`, `set_rows`
  collapses to one upload), or E028 for the deep-context case.

### P8 - GDN at shallow depth (measurement first)

- **Question:** what share does the 36-layer linear-attention path take at d4096, where the box's pp is
  best?
- **Why:** on the dev-box dummy at prefill width `gated_delta_net_cuda` is 19.0% of device time and
  3.1 ms per call, the single largest kernel there; in A5 (d131072) it is only 3.5%. Nothing local says
  which of those two the box resembles at shallow depth.
- **Next step:** the section 1 trace, family table at d4096.

### P9 - the meta buffer's per-tensor init is what `graph:alloc` is made of

- **Measured 2026-09-28 (`bee859d04`).** The split-pass hypothesis is falsified: `sched:split` is 0.88 ms of
the box's 24.10 ms `graph:alloc`, and `sched:init_nodes` is 22.25. Phase split per meta init, from the
`meta:init_*_ns` counters: **prep 1.65 us of 2.94 us** (container lookup plus split state), tail 435 ns,
create 507 ns, srcs 349 ns, over ~6800 inits per ubatch. Dev box with 1 sub-buffer: 1.94 / 431 / 170 /
185 ns, so only create and srcs scale with the device count. Total 20.0 ms/ubatch, **4.1% of prefill**,
pure host work with no numerics in it.
- **Why prep is 1.65 us.** `ggml_backend_meta_get_split_state` caches on the tensor *pointer* plus a
  `memcmp` of the whole stored `ggml_tensor`, and on any mismatch it **clears the entire cache**
  (`ggml-backend-meta.cpp`, around the `split_state_cache` in `calculate_split_state`). A rebuilt prefill
  graph reuses arena structs, so the first mismatch of each ubatch drops everything and every later tensor
  recomputes.
- **Two fix candidates, cheapest first.**
  1. ~~Evict the one stale entry instead of clearing the cache.~~ **Ruled out as unsound 2026-09-28.** The
     split state is a pure function of the *sources'* states (`handle_generic(src_ss)`, `handle_flash_attn_ext(src_ss)`,
     `handle_gated_delta_net(src_ss)`, ...), so an entry whose own bytes still match can be stale because a
     source's shape changed under a stable pointer. The whole-cache clear is what invalidates those
     descendants; per-entry eviction would serve a wrong state and the PPL gate would have to catch it.
  2. **Cut the lookup cost, keep the invalidation.** Roughly 6-8 of the 11 ms/ubatch is the lookups, not the
     recompute: one `get_simple_tensor_container` plus 5.97 source lookups per init (from
     `meta:init_src_remaps`), each a `std::map` find on a pointer-pair key over ~6800 entries plus a
     400-byte `memcmp`. A per-graph generation counter plus an `unordered_map` keyed by pointer removes the
     `memcmp` and the mid-graph clears and makes the lookups O(1), with states still recomputed once per
     tensor per ubatch. Sound, smaller change, ~6-7 ms/ubatch.
  3. **Content-keyed memo** `(op, type, ne, nb, view state, sources' state hashes)`: the only variant that
     survives a rebuild, so the recompute goes too. ~10-11 ms/ubatch, more code in the delicate part of the
     file.
- **Next step:** 2 first; 3 if the leftover recompute still shows in `meta:init_prep_ns`.
- **P9b implemented and validated on the dev box, 2026-09-28.** The cache is an `unordered_map` keyed by
  (tensor pointer, `assume_sync`); the 400-byte tensor copy, the `memcmp` and the whole-cache clear are
  gone, and the states for a tensor are instead dropped in
  `ggml_backend_meta_buffer_init_tensor`. That entry point runs for every tensor newly allocated for a
  graph - nodes, views through `ggml_backend_view_init`, and reshaped input leafs - and never for static
  weights, whose states therefore stay cached. Correctness rests on the walk order: a tensor's sources are
  initialized earlier in the same walk, so their states are recomputed before they are read.
- **Gate:** PPL with `-sm tensor` on the sparse corpus at `-c 8192` is **bit-identical before and after**
  (270867.5339 / 270355.0350 / 266980.6971). Note `-sm tensor` shifts the digits by itself, against
  266980.7507 for the default split, so a meta-path gate has to compare like with like.
- **Result, dev box, same command:** per init 2.72 -> 1.87 us (prep 1.94 -> 1.22 us, tail 431 -> 304 ns,
  create and srcs unchanged), `sched:init_nodes` 20.49 -> 17.71 ms/call.
- **Result, box run 2026-09-28** (`results/user/llama-bench/3d8452b0f/run-prof.log`, 6,793 inits per ubatch):
  per init 2.94 -> 2.32 us (prep 1.65 -> 1.11, tail 435 -> 315 ns, create and srcs unmoved), i.e. 20.0 ->
  15.8 ms of the 489 ms ubatch, and **`pp4096 @ d16384` 2070.90 -> 2076.09 +- 2.40** (+0.25%). The host
  saving is ~0.9% of the ubatch and the wall sees 0.25%, so pass-through is roughly a third - the same ratio
  E077 found, and a reminder that host-side items have to be priced through it. `sched:init_nodes` reads
  21.26 against 22.25 ms/call; the two `erase` calls per init sit outside the timed phases, so they are in
  that region but not in the counters.
- **Left on the table, box numbers:** prep is still the top phase at 1.11 us of 2.32 (48%), and it is the
  per-graph recompute plus a `std::map` find in `stc.simple_tensors` (once per init and once per source
  lookup). Switching that container to an `unordered_map` is the same trick again and cheap; P9c's content
  key is what removes the recompute. Tail 315 ns is the zero-slice scan plus the insert.
- **Bound, added after the first version.** The map is capped at `split_state_cache_max` (2^16 entries) and
  cleared when exceeded. Keys are addresses that are never dereferenced and the split state holds no
  pointers, so a stale entry is at worst a wrong number, and clearing is always sound; the cap only gives a
  long-lived process a ceiling. It does not fire at the observed sizes (27k inits per pass, ~9k entries).
- **Validation:** pp t/s at d16384 plus the `meta:init_prep_ns` and `sched:init_nodes` rows, and the golden
  PPL - the split state decides per-device slicing, so a wrong state would move the numbers.

### P10 - overlap the build and alloc with the previous ubatch's device work

- **Question:** can ubatch k+1's `graph:build` and `graph:alloc` run while ubatch k is still executing on the
  device, so the host path stops being additive?
- **Why it might work:** nothing in the build for k+1 needs device results, only the memory context's
  bookkeeping (n_kv, the pool state), which is host-side and already known. The two `gf_res_prev` slots are
  machinery for holding two graph results at once; today they alternate output classes rather than pipeline.
- **Prize:** the 22 ms/ubatch (4.5%) without touching a single shape, so it does not inherit P2a's problem.
  If host time is already hidden behind device work, this measures that too.
- **Cost and risk:** a change in `llama_context::decode`'s loop and in the ordering guarantees around
  `mctx->apply()` and `set_inputs`, the same class of hazard as E073's synchronize-before-set_inputs
  invariant. Two graphs in flight also means two copies of the input staging; the sched has `n_copies` for
  that, so the buffers exist.
- **Deciding metric:** `phase:prefill` wall plus the region table at pp4096 d16384, with `graph:alloc` and
  `graph:compute` no longer back to back.

## 7. Not to re-open

- E058's closures: no row cache, no `POSIX_FADV_RANDOM` change, no worker divisor, `-lzm off` is not a
  residency test.
- E073's invariant: the scheduler is synchronized before `set_inputs` when the graph is reused; the
  fusion stays on, and `GGML_CUDA_DISABLE_FUSION=1` is a diagnostic only.
- H26 is a decode item: ~4% of decode device time at 519 launches per step. Prefill's share is 2.1%
  (5.5 above) and no A/B can resolve it.
- Decode-only levers stay in `decode-comms-plan.md`.

## 8. Traps

- Region totals are inclusive (`graph:compute` contains `meta:*`), so never sum the region table into a
  wall.
- `rocprofv3 --stats` may replay kernels: use shares from that instrument, never absolute ms.
- The dev-box dummies are F32 and small: their conversion and dequantize launches do not exist on the
  real model, and their device shares do not transfer. Node counts and launch *structure* do
  (6873/1372 against the box's 6825/1372).
- `ggml_type` 6 is Q5_0 and 12 is Q4_K; a row reading `(ggml_type)6` is Q5_0, not Q6_K.