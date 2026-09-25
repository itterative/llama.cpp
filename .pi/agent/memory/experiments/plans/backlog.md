# Backlog - candidate threads, qwen4exp on RDNA4

Not experiments yet. A thread becomes an `E<nnn>` id only when it has a falsifiable hypothesis and a
deciding metric (see `../PROTOCOL.md` 3.1).

Anchors at commit `ebbb18522`. Markers: **[v]** read directly by me, **[s]** from the `scout-1` survey
and not re-read, **[x]** corrected after verification. Backend picture lives in
`../../rdna4-rocm-build.md`; real dims in [model-shape.md](model-shape.md).

Format: one item per `###` heading, `id - question`, with the fields as bolded labels underneath. A
struck-through id means the thread is answered or dead; the answer stays inline, because these notes
are why later experiments were scoped the way they were.

**Live right now:** H9's prefill question (the pool works, and excluding prefill from it is a workaround
nobody likes), H18 (the MTP tax, now framed as over-drafting), H17b (is the chain 4x replicated), E008b
(the measurement that decides the whole PLE/prefetch line), and the H13 leftovers.

---

## Prerequisites

### B0 - make the build un-shadowable

- **Why:** `BUILD_SHARED_LIBS=ON` plus `~/.local/lib64` holding a Sep 15 llama.cpp means an unpinned
  run silently measures old code. `GGML_STATIC` is a hard `FATAL_ERROR` on the HIP path, so rpath is
  the fix: `CMAKE_BUILD_RPATH=$PWD/build/bin`. Found in E001.
- **Status:** open, and still relevant - this session lost several runs to it again. The current
  workaround is `LD_LIBRARY_PATH=$PWD/build/bin` plus an `ldd` check before every measurement.

### B1 - `test-backend-ops` on AMD GPUs

- **Status:** closed on the dev box. It does not crash: 1500/1500 non-FA and 3973/3979 FA cases pass on
  gfx1201 / ROCm 6.4.4, so rule 3 has a working gate (E001 update). Re-baselined on 7.1.1 in E019:
  **5633 OK / 7 FAIL**, all the known FA 192/128 family; capture works here now; `-j 1` only.
- **Still open:** the bench-box hang the user remembers is real but box-specific - candidates are the
  4-GPU config, its ROCm version, or its code state, all B2 unknowns.
- **Two sharp edges:** no insufficient-memory skip logic, so an oversized case OOM-aborts `[s]`; and
  `-b` is an exact `strcmp` that exits 0 having tested nothing on a typo `[v]`.

### B2 - bench box facts still unknown

- ROCm version, PCIe topology, system RAM. The gfx target is now known: **gfx1201 on both boxes**
  (user-confirmed), so a dev build is ISA-valid there. What remains decides whether numbers are
  comparable, not whether binaries run.

### ~~B3 - dev-box ROCm version policy~~ resolved: upgraded

- The user moved the dev box to Fedora 44 / ROCm 7.1.1 on 2026-09-17, so the predicted comparability
  break is real: **every v1 dev number (E001, E002) sits on the old stack**, and hw profile v2 must be
  re-established before dev results are usable. Predicted upside now testable locally: graph capture
  (never succeeded under 6.4, works on the bench) and FA family selection pp vs decode.

### B5 - rebuild the dev tree against ROCm 7.1.1

- `build/` was unloadable, verified: its `libggml-hip.so` still `NEEDED` `libamdhip64.so.6` /
  `librocblas.so.4` / `libhipblas.so.2`, all deleted by the upgrade, so the `LD_LIBRARY_PATH` pin cannot
  save it. Flags are unchanged (the prefix `/usr/lib64/rocm` is the same). Hazard: the upgrade also
  removed `compiler-rt18` and `libomp18`.
- **Status:** presumably done - E019 re-baselined the ops gate on 7.1.1 on this box, which required it.
  Close formally on the next clean configure.

---

## Likely bottlenecks (this is where the effort should go)

### L1 - is the Q5 n-gram table actually resident?

- `-lzm auto` lazy-reads any tensor over 4 GiB (`src/llama-model-loader.cpp:1088-1100`) `[v]`, so a
  ~33 GB table is probably served from the page cache on demand, not held in RAM.
- **Next:** check the load log for `lazy read enabled`, then A/B `-lzm off` / `auto` /
  `--load-mode mmap+mlock` on `tg` stability. Zero code change, no rerun to diagnose, and irregular
  `tg` over a 20 M-row random gather is exactly its signature.

### L2 - PLE placement is *suspected* of costing tg, unresolved

- The table is the `ggml_get_rows` source at `src/models/qwen4exp.cpp:1202`. On CPU that is host gather
  + H2D copy + split, and the n-gram hash is already host-side (no int64/xor in ggml).
- **Whether that is what costs 35 ms/token is unknown** - the user's own read is "hard to say". Putting
  the table on GPU is not an option (mirrored, ~30 GB/card), so the earlier "keep it on GPU" framing
  here was wrong.
- **Next:** measure faults, do not build. `ple-prefetch.md` method 1 = `iostat` during a decode run,
  zero code. See E008b.

### L3 - 10-of-512 expert routing on HIP **routing is fine; E053 redirected this row**

- `num_experts 512`, `per_tok 10`, `moe_intermediate_size 640`: ~2% of expert weights touched per token
  per layer, so `tg` is a scattered-read problem.
- Good news from the survey - **corrected by reading the dispatch code** (see the `rdna4-rocm-build`
  memory, "MoE matmul dispatch at decode"): `should_use_mmq` returning true on RDNA4 and MMQ tile
  selection being expert-aware (`mmq.cu:248-251`, `:380-386`) are **prefill facts, not decode ones**. At
  batch 1 `ggml_cuda_mul_mat_id` returns into mmvq first (`ggml-cuda.cu:1993-2001`), because batch 1 is
  below every RDNA4 per-type cap (`mmvq.cu:258-282`), so decode runs mmvq's nwarps table - and the ~115
  GB/s is on that table's *tuned* 8-warp branch. The trace agrees: there is no `mul_mat_q` row.
- **E053 first read, then partly retracted.** The quantized weight matvecs are 41.3% of decode device time
  on 4 cards (519 launches per card per step, 7.05 ms per card per step), which made this look like the top
  item. It is not the routing: 519 launches cannot be 24,576, so we touch the 10 active experts and nothing
  more. What the row actually measures is **weight reads at n_rows 1 running near 20% of achievable
  bandwidth** (0.8 GB per card per step should be ~1.3 ms and takes 7.05). That redirects the question to
  mmvq's RDNA4 configuration (H6: `mmvq.cu:417-492` nwarps whitelist, `ggml-cuda.cu:1512`
  `prefer_f32_output`) and to the 519 per-matvec `quantize_q8_1` launches (0.58 ms, ~8% of the matvec time
  re-quantizing the activation once per matvec).
- Rows by type, for whoever picks this up: 6 = **Q5_0** (20.4% of the visible total, 29 us per call - and
  Q5_0 rather than Q4_K is exactly the block-size fallback this model's 640-wide `ffn_down` triggers; see
  the `quant-block-size-fallback` memory), 8 = Q8_0 (11.5%), 12 = Q4_K (8.8% across both variants),
  14 = Q6_K (4.2%). Whether the Q5_0 row really is `ffn_down` is still the open identification - the
  loader's type census answers it.
- **Cheapest new candidate, now answered (E054): mmq is not the fix.** Forcing `MUL_MAT_ID` onto mmq at
  batch 1 costs 3.6% of tg on the dev box (1.2% of that is the glu fusion the same knob disables), so
  mmvq's tuned branch is the right kernel at n_rows 1 and the ~115 GB/s is an mmvq-internal question -
  `calc_rows_per_block` / `small_k`, or the `-sm tensor` k-split below. What is left of the idea: the
  k-split case was excluded on purpose, so it is still untested on 4 cards.
- **E055 found where the bytes actually go, and it is confirmed on both boxes**: mmvq's `small_k` shape is
  blanket-excluded for RDNA (`should_use_small_k`, `mmvq.cu:~1090`, no comment) and `calc_rows_per_block`
  omits `MMVQ_PARAMETERS_RDNA4`, so an 8-warp block reduces one 480-byte `ffn_down` row at a time. Letting
  gfx12 take small_k (`GGML_CUDA_MMVQ_RDNA4_SMALL_K=1`) is **+4.1% / +7.4% tg on the dev dummies and
  +5.5..6.6% on the real model across 4 cards at every depth**, pp unmoved, golden bit-identical, ops green.
  Flat in depth, so it stacks with the pool rather than overlapping it. Full record:
  [runs/E055-mmvq-small-k-rdna4.md](../runs/E055-mmvq-small-k-rdna4.md).
- **Open on this item**: the default is now on (`855a65544`), so **every tg number on this branch before
  that commit sits ~6% low** - re-baseline E050/E052/E043 comparisons rather than reading them as movement.
  Still to decide: whether it is worth an upstream proposal, which needs an RDNA3 data point and the
  `has_ids` / `should_halve_iters` objection answered in advance. pp on the box is also unconfirmed (E055
  ran `-p 0`); locally it did not move, and batch 4096 is mmq territory so it should not.

### L4 - 16 GB of fp32 recurrent state

- `mamba_ssm_dtype: float32` x 36 linear-attention layers with key 2048 / value 6144 wide. State size is
  a fixed VRAM tax that competes with the 262 k context, and it resists fp16 tricks. Note that
  `n_rs_seq` multiplies it again - see H18.

---

## Closed by the survey (do not re-probe)

### ~~P1 - which ops fall off the GPU~~ measured

- `test-backend-ops support -b ROCm0` over 13 families -> 9906 supported / 2384 unsupported cases, and
  the non-FA `test` run passed 1500/1500 `[v]`. Every op in the graph has a real HIP kernel.
- Remaining escape hatches: elementwise contiguity gates, and `ARGSORT` needing `ne[0] <= 1024` `[v]` -
  real `num_experts = 512`, so the router stays on GPU.

### ~~P4 - which FA family does gfx1201 get~~ `mma_f16` for prompt processing, not for decode

- `amd_wmma_available` + DK 256 + `gqa_ratio_eff 4` gives threshold `Q->ne[1]*4 > 16`
  (`ggml/src/ggml-cuda/fattn.cu:667-671` `[v]`, `(256,256,*)` instances exist `[v]`). Decode of 1 token
  falls to tile/vec. Since sparse FA lives only in `mma_f16`, H4b is a **pp-only** win - which is where
  long-context cost is anyway.

### ~~H3 - HC kernels may be shape-restricted~~ not on HIP

- The gate is dtype-only, all-F32, no shape restriction (`ggml-cuda.cu:5492-5501`) `[v]`. The
  `ne[1] == 4` rule I feared is Metal/Vulkan's. Replaced by N1 below.

### ~~P5 - does the model fit~~ capacity is not the constraint on this box

- See `model-shape.md`.

---

## New, from the survey

### N1 - `RMS_NORM+MUL+ROPE` fusion is rejected for qwen4exp because it is IMROPE

- The 3-node fusion accepts only `GGML_ROPE_TYPE_NORMAL`/`NEOX`
  (`ggml/src/ggml-cuda/ggml-cuda.cu:2734-2735`) `[v]` and `llama-model.cpp:3063-3070` gives this arch
  `IMROPE` `[s]`. Self-contained, clearly-scoped fusion gap on 12 full-attention layers.

### N2 - a quantized KV cache may not be usable

- `SET_ROWS` dst whitelist excludes **Q4_K/Q5_K/Q6_K** `[s]`, and qwen4exp writes KV through
  `cpy_k`/`cpy_v` = `set_rows`. If anyone tries a Q4_K KV cache to fit 262 k context, that is a support
  failure, not a slowdown.

### N3 - `GGML_CUDA_DEVICES` for multi-device testing

- `[x]` **does not work on this box.** It exposes N *virtual* devices round-robined over physical GPUs
  (`ggml-cuda.cu:235-259`) `[v]` and takes a *count*, but requesting more devices than exist dies with
  `invalid device ordinal` on HIP, so `-sm layer` split behaviour and split counts stay T2-only here.
- What *is* usable on 1 card: `-sm tensor` still routes through `ggml-backend-meta.cpp` with
  `n_backends = 1`, which is how the meta-backend crash (E044/E045) was reproduced locally at all.

### N5 - HC inputs are F32-only

- The gate requires f32 for all HC operands `[v]`, so any fusion or cast that lands f16 there silently
  leaves the fused path. Constraint on H2, not an experiment.

---

## Hypothesis-ready (write the record, then run)

### H1 - ~~`-sm tensor` unavailable for qwen4exp forces layer split~~ revised by E005

- This branch *throws* for `-sm tensor` upstream: `llm_arch_supports_sm_tensor` returns false
  (`src/llama-arch.cpp`) and `llama_model_create` raises `LLAMA_SPLIT_MODE_TENSOR not implemented`
  (`src/llama-model.cpp:358`). **But the user runs tensor split on the bench box**, so their fork
  enables it - and merging forward will break their command line until that patch comes along (B4).
- The upstream guard is test-driven (`// TODO: fix test-llama-archs`), i.e. the blocker is the dummy
  model, not the backend. This row previously claimed layer split was forced; it was not, and that
  error cost a detour.
- **E006: tensor wins tg, as the user expected.** For pp the general rule is the *opposite* - layer
  usually wins, because layers pipeline across cards and the transfers hide behind inter-layer compute
  overlap - so this run's layer-loses-pp result is an anomaly, not a rule, and it is now read as a
  symptom of the CPU split (F3, E007b). The guard stays an obstacle to clear in B4.

### H2 - the hyper-connection chain is under-optimised on HIP

- HC replaces every per-layer norm, so it is per-layer and every-token in both modes. Only `_PRE`
  (gated) and `_POST` (comb=null) are emitted `[x]`; both are f32-only `[v]`; and `rms_norm+mul` fusion
  was only just enabled (`41abbfd59`). New kernels are usually correct before they are tuned. Bounded
  by N5.

### H4a - stop paying the indexer + mask-rebuild tax while compaction is unavailable

- Mask rebuild is `fill(-INF)` + `set_rows` + `add` per full-attn layer per ubatch
  (`src/models/qwen4exp.cpp:735-758`) `[v]` - traffic scaling with context for a mask whose interior the
  kernel ignores. `:566` already trims the upload to `1/ratio` of cells. Obsolete the moment H4b lands.
- **Corrected by E024: H4b landed and this tax is small** - the mask tensors are `O(n_kv)` f16
  (~0.3 MB/layer/step at 40k), while the indexer gather and pooling next to them are ~50 MB.
  Deprioritised in favour of H9; the per-ubatch cost that matters at prefill is `O(n_blocks x n_tps)`,
  which is host-side (E021).

### H4b - port the mask compaction to HIP, then flip `n_kv_max` **done by E020**

- Narrowed by the survey to: one warp-ballot kernel (`fattn.cu:10-89`, `WARP_SIZE == 32` which gfx1201
  has, but `ggml_cuda_pdl_*` are NVIDIA-only), the `#if !defined(GGML_USE_HIP)` compile guards
  (`:92-96`, `:109-113`, `:133-140`) `[v]`, and the call site (`qwen4exp.cpp:767`) `[v]`.
- **Updated by E001: flipping the call site first is inert, not a safe first step** - sparse cases
  already report SUPPORTED and compute dense, so results and cost are unchanged either way. Effect
  size: `indexer_budget 2048` of 262,144 context, on 12 of 48 layers, pp-only per P4.
- **Result (E020):** pp512 +23.3% @40k / +58.5% @164k, tg flat on the pre-rtile build (superseded by
  E043: with rtile + block selection the sparse arm is ~1.4 ms/token/GPU cheaper than dense at 131k
  decode), numerics match dense to 6e-7. P4's list was short three items: the `__ballot_sync` 64-bit
  mask signature, the `may_use_sparse` DKQ whitelist, and the fact that RDNA has no FA device code below
  16 tiles so the tiling must be 1x16 and had to be added to `generate_cu_files.py`.

### H5 - PLE n-gram hashing

- `ple_n_heads = (3-1)*8 = 16` gathers per token from a ~20 M x 2560 table = ~51 B params = 28% of the
  model, serving one layer, hashed host-side. Superseded in priority by L1/L2.

### H6 - fp32 output preference on RDNA4

- `prefer_f32_output` is forced on for RDNA4 (`ggml-cuda.cu:1512`, `:1514`) `[v]`, and `mmvq.cu:417-492`
  has an RDNA4-only `nwarps` whitelist `[s]`. Confirmed as real, still unmeasured: a per-model override
  is a plausible small win.

### H7 - VMM disabled

- `GGML_HIP_NO_VMM` defaults ON, and the `VMM: no` banner is just that flag, not a device query `[s]`.
  Allocation/fragmentation behaviour differs from a CUDA default - relevant to both the 16 GB box and a
  4-card split.

### H8 - `GGML_HIP_RCCL=OFF`

- The "rebuild with NCCL" warning is `#ifndef GGML_USE_HIP`, so without RCCL the fallback to internal
  AllReduce is **silent** `[s]`. Only worth a record once B2 reports the topology.

### H9 - cache pooled indexer block keys (coarse cache) **shipped as a long-context decode feature**

- **Original question (added by E024):** `build_qsa_top_k` re-gathers the whole raw indexer cache and
  re-pools it every step - ~76 MB and ~33 graph nodes per QSA layer per step at 40k, about 2x dense
  attention and ~19x sparse attention post-E020, and unlike the host fix this *is* per layer, so 12x on
  the real model. Block keys are immutable once a block's `r` tokens are written, so a coarse cache
  deletes the gather (~21 MB) and the `r` pooling passes (~29 MB). Why it was not a small change: block
  membership is a function of *position*, and `seq_add` / `seq_div` (context shift) rewrite positions,
  so it needs an `O(n_kv)` rebuild path anyway plus a new tensor threaded through
  `seq_cp`/`seq_keep`/eviction/defrag/`n_pad`/state save-load in `llama_memory_hybrid_idx`.
- **Prize, measured by E042+E043 (decode-only traces at 131k):** the whole family - gather, 4 rect
  copies, pooling adds, scale, norm, rope - is **11.1 ms/token/GPU**, 84% of the chain, 35% of the qsa
  arm's device time, and it is output-neutral (block keys are static once a block completes; block start
  positions never move). E037's 36% was wrong - it excluded norm and rope as "position-dependent" - and
  E025 is obsolete: the prize is now measured.
- **Built:** `d8bce4e25` (E044), gate `Q4EXP_POOLED`, default off, bit-identical to the gate-off build on
  every dev-box test. Shape as designed in [h9-pooled-block-keys.md](h9-pooled-block-keys.md): a
  mirrored f32 `[idx_dim, n_blocks, n_stream]` pool tensor inside the existing idx cache, written in the
  graph that writes the raw keys with existing ops, watermark + recorded-run invalidation, today's chain
  as the rebuild and the fallback. 491 lines across 6 files, no new kernel, no new file.
- **What the work taught us about invalidation:** the surface is the wrapper's own seq ops plus one
  branch in `apply()` - smaller than feared - but the correctness traps were elsewhere. An input with no
  consumer gets no host buffer (NULL `data`, SIGSEGV), and the rebuild write-back must target the pool,
  not the chain tensor it just read (E044 findings 4 and 6). A third trap was operational rather than
  arithmetic: speculative rollback arrives as a checkpoint `state_read`, not a partial `seq_rm`, so
  clearing the run there re-derives everything on nearly every step - fixed by clamping the watermark to
  the surviving blocks (E045, `90b9ccf9d`).
- **Answered (E049-E052):** the pp regression was never the pool's maths - `sched_reserve` measured an
  empty cache, so the first pooled ubatch re-reserved the compute buffers and every depth-proportional
  tensor kept ratcheting. Fixed, confirmed on 4 cards (pp back to -0.5..-0.9%, tg win intact), then
  collapsed to one graph shape and widened to all ubatch widths, with prefill behind
  `Q4EXP_POOLED_NO_PREFILL`. What the pool is for: long-context decode, +30% tg at 131k.
- **Open, and the live one:** `Q4EXP_POOLED_NO_PREFILL` is a workaround that works, not an explanation.
  The user's position is that prefill should not have to be excluded and something else is at play, which
  is fair: E051 showed pooled prefill neither gaining nor losing once the topology confound was removed,
  and E052 showed it costing 2.9% with 24+7 re-reservations - but those reservation ms are not additive
  cost (their bulk is a device sync that would otherwise be counted inside `graph:compute`, see E049's
  review note), so what actually accounts for the 2.9% is unidentified. Reading it as "pooling prefill is
  worthless" is a measurement, not a mechanism.
- Also open: the pool taxes **354 MiB/card at ctx 245760**, so f16 storage (H15/P3) is still on the table;
  and the crossover depth where pooling stops paying (-5% at 4096, +2.6% at 40960 on 4 cards) is
  unexplained.
- **Scope:** prefill, decode, 4-card.

### H10 - mmq cutoff tuning for MoE models

- **Added by the user after E026.** Their RDNA3/4 mmq retune has not fully landed, and the dummy showed
  pp512 -11.2% at 40k from it while decode was flat. The `mmq` cutoff/selection that decides when a
  quant matmul takes the tensor-core path is not adjusted for MoE shapes (many small expert GEMMs at
  large batch), and upstream does not have those changes yet. Track separately from the sparse/QSA line.

### H11 - prefill runs the GPUs at 20-30% while decode sits near 100%

- Reported by the user on the bench box, present before *and* after the rebase, so it is not a merge
  artifact. This is inverted: prefill is the compute-bound phase (512-8192 tokens per ubatch through 48
  layers of MoE GEMMs) and should be the one saturating the cards, while decode is small-kernel and
  collective-bound.
- **Three readings, in the order I would kill them:**
  1. *The 100% on decode may not be work* - `amd-smi`/`rocm-smi` utilisation counts any resident kernel,
     so a spin-wait custom/p2p AllReduce reads as busy; E005 measured **9% util during decode** on this
     same box, which is either the opposite truth or a different sampling window, and both cannot be
     right.
  2. *Prefill may be host-serialised between ubatches* - per-ubatch host work exists (`set_input_qsa`
     bias fill measured at 4-6 ms per ubatch at 25-30k on the dev box, E021; input construction and the
     QSA mask rebuild), but 512 tokens at pp512@131k is ~1.3 s of wall time, so a few ms of host work
     cannot hide 70% of the card.
  3. *Something in the pp path is genuinely GPU-idle* - cross-card sync per layer with tensor split, MoE
     expert dispatch with a device->host count read forcing a sync, or graph rebuild when ubatch shapes
     change (E010: one build is ~20-22 ms; pp changes n_kv as the cache grows, so a reuse miss would
     stall).
- **Superseded in likelihood by the n-gram streaming hypothesis**, which the arithmetic supports: the hw
  profile has the PLE table as Q5 in **system RAM** at ~30-36 GB, and the box has **62.7 GiB** (the
  "32g" in the profile name is not RAM - corrected after the screenshot), so the table can be cached but
  the 111 GB file cannot - it is served by page-cache misses from SSD, and the table ranges deliberately
  get `POSIX_MADV_RANDOM` (read-ahead off) with `MAP_POPULATE` skipped and no `WILLNEED`
  (`ple-prefetch.md`, F1). That predicts exactly the asymmetry: prefill issues thousands of scattered
  single-page reads per ubatch and starves the cards, while decode does 16 lookups per token and mostly
  hits what is warm. E017's A/B is the same effect at smaller scale: 35.2 GB table gave pp4096 6017 t/s,
  3.7 GB gave 6407 (**+6.5% pp**) with **tg unchanged** (182.14 vs 182.40). Caveat: llama-bench may be
  *amplifying* it (E030).
- **Screenshots** in `results/user/prefill-ssd-probable-issue/` (gitignored, local-only) support the
  paging story rather than replacing it: all four GPUs at **26% util** at the same time, **~70 W** each,
  and **load average ~1.2** on a many-core box, so neither the GPUs nor the CPU are working - the process
  is waiting. VRAM 22 of 32 GiB per card, so weights are resident and this is not a GPU memory problem.
  The disk panel shows the root device busy at **~66 MB/s of reads** (`▲` is read; the `swap 7.00 GiB`
  row is the swap *size*, not usage, and no swap-in was observed). The rate is the useful number: 66
  MB/s is nothing for an NVMe device, and at 4 KiB pages that is roughly **16k reads/s**, i.e. the device
  is latency/IOPS-bound on random single-page accesses, exactly what `MADV_RANDOM` over a scattered
  n-gram gather produces. That puts F1 back in play with a concrete mechanism: prefill knows every token
  of the ubatch up front, so the row set is predictable and `WILLNEED`/`readahead` can coalesce what is
  now ~16k independent page reads/s. `[x]` An earlier reading of the same screenshot took the arrow as
  writes and the swap size as usage, and concluded "box is over-committed"; that version is dropped, but
  whether any of it is swap is still cheap to settle (`vmstat` `si`/`so`, majflt/s), which is what E008b
  asks for.

### H12 - put the PLE table in VRAM instead of streaming it **dead on cost/benefit**

- The structural version of H11. The table is 32.8 GiB (Q5_0, 51.2 B params) and 29% of the file, and it
  is mirrored across cards, so it cannot be placed on GPU as it stands (E007). Split 4 ways it is
  8.2 GiB per card against ~10 GiB currently free, so it *would* fit - the size is no longer the
  obstacle, E031 is: once `-lzm on-direct` removed the demand faults, the table costs 1760 bytes of
  reads per token, which is nothing. Kept as the record of what the streaming was.
- E014 (resident table, `-lzm off`) is the variant that is now feasible and worth a flag-only run,
  because 32.8 GiB fits in the box's RAM.

### H13 - select QSA at block level, then expand, as the paper does **implemented**

- Measured on the dev box (E039): **+6.3% tg / +6.8% pp at 164k**, +2.3% / +2.0% at 40960, flat at
  8192; deep-arm PPL +8e-8, shallow golden corpus bit-identical, ops gate green. Same-binary A/B through
  the temporary `Q4EXP_CELL_SEL` gate.
- What it deletes: the `n_kv` expand gather, both `cont(permute)` copies, the f32 per-cell mask add,
  top-k over `n_kv` (11 dependent launches, 4x less input now), and `cell_blk` with its O(n_kv) host fill
  since the expand was its only consumer.
- **Bench (E040):** tg +7.2% and pp +22-23% at 131k, but that delta also contains the rtile decode path
  and the per-build `n_kv_max` fix, and rtile cannot move prefill, so the pp share is not attributable
  to H13 without one more run at the same build with `Q4EXP_CELL_SEL=1`.
- **Leftover:** once the bench conclusion lands, the gate and the per-cell path come out (the `!blk_bias`
  fallback is a different thing and stays). Also still owed: the selection-set differential (old vs new)
  and a live multi-seq run on the real checkpoint.
- **Scope:** prefill.

### ~~H14 - is the model dispatch-bound at decode?~~ closed by E038: no

- The chain's device time is 12.2 ms/token at 131k against the 15.5 ms/token measured gain from
  `Q4EXP_NO_INDEXER`, so 79% of the effect is kernels, not launch gaps. The earlier 612-launches/token
  figure was wrong twice over (3 chain builds per token, and per-call counts normalized across phases).
  Keep only as a footnote: graphs-off costs ~7% of tg on the dev box (E008).

### H15 - keep the indexer gather and pooling in f16 **superseded by H9, ceiling cut by E053**

- **E053 sizes what is left**: the entire sparse chain (mask->indices, rtile FA, combine, radix top-k) is
  **3.2% of decode device time**, against the 35% E042/E043 measured pre-pool. Halving the one f32 read
  that survives is a ~1% prize, not a 35% one. Keep the VRAM argument (354 MiB/card at 245760), drop the
  perf argument.

- In the `CACHED` variant the gather, the pooling, the norm and the rope are gone, so the only f32
  traffic left is the pool read under the score matvec. The remaining version of this item is the plan's
  P3: store the pool as f16, which halves that read (0.59 -> ~0.3 ms/token/GPU) but rounds the cached
  key, so selection can move last-bit and the golden would need re-baselining. Keep the original note for
  the `REBUILD`/fallback path, where the gather is still f16 -> f32.
- **Scope:** prefill, decode.

### H16 - 4-card decode comms: in-tree one-shot allreduce, +2.5% at depth, tied at 4k **done, opt-in**

- **Resolved by [plans/decode-comms-plan.md](decode-comms-plan.md)**, which has the arms, the cost model,
  the correction below and the three defects it took. `GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal`
  reaches it through `auto`: paired at r=10 it gives tg128 30.97 vs NCCL 30.20 at d131072 (+2.5%) and a
  tie at d4096, host cost per collective 14.8 -> 6.3 us, pp unchanged.
- **Correction worth keeping where it will be read:** the first reading of this was +7.2% / +8.1%, from
  unpaired `-r 3` runs in separate invocations. Paired and alternating at `-r 10`, it is +2.5% at depth
  and zero at 4k. Same class of error as E055's first 4-card A/B, and the noise floor that exposed it was
  already in my own table (two arms differing by 3% where the code path was identical) and I applied it
  only to the result I liked.
- Still open on this item: the 256 KiB size cutoff is a guess, `auto` does not pick one-shot so the path
  needs three env vars, the inboxes reserve 4 x `GGML_CUDA_AR_DIRECT_TMP_BYTES` (256 MiB) per GPU, and
  whether `GGML_CUDA_ALLREDUCE` should stop defaulting to NCCL on this branch.
- **RCCL itself has nothing left.** `NCCL_ALGO=Tree` is silently ignored (`AllReduce | Tree = 0.0/0.0` in
  its own tuning table), `NCCL_ALGO=FC` is accepted and 8% slower, LL and one channel are already
  auto-selected at 10240 B, and `VMM: no` on all four devices means cuMem/symmetric windows cannot exist.
  `RCCL_USE_AMD_SMI_LIB=1 NCCL_CUMEM_ENABLE=1` is a **hazard**: the collective got cheaper and tg fell to
  3.60. Do not re-run that pair.
- **Scope:** decode, 4-card.
- **History that motivated it:** E037 measured `ncclDevKernel_Generic_4` at 163,124 calls / 22.0 s for the
  chain-off arm, 96 collectives per graph build at ~135 us each for a 2560-wide f32 activation, identical
  with QSA off; E038 pushed the per-call number to 485 us at 131k, i.e. peer-wait growing with depth. Both
  are now explained by the API-call cost model in the plan rather than by NCCL being slow at bytes.

### ~~H17 - is the QSA selection global or per-device under `-sm tensor`?~~ no correctness bug

- `src/llama-model.cpp:511-514` maps `cache_idx_(k|v)_l*` to `SPLIT_AXIS_MIRRORED` ("the qsa indexer has
  one key head and its projections are mirrored, so its cache cannot be split") and that rule came with
  upstream's own qwen4exp commit `6c84c7d5d`; the bench load log confirms it (main cache 198.00 MiB
  logical vs a 49.50 MiB buffer = quartered, indexer cache 24.75 MiB logical vs a 24.75 MiB buffer =
  whole), so every device sees all cells and the top-k is global, as the bench probe line showed
  (`nkv=4352`, `gqa=12`). The mirror costs ~384 MiB/card at 131k.
- Note also that `-sm tensor` for this arch is a fork-local `tmp:` change: `a8b24dfdf` deletes
  `case LLM_ARCH_QWEN4EXP: // TODO: fix test-llama-archs` and adds
  `ggml_build_forward_expand(gf, res_hc)` to pin `hc_init` into layer 0's graph split - so every bench
  number in this directory was measured with that workaround in place.
- **Cost half is open as H17b.**

### H17b - is the QSA chain replicated into all four device sub-graphs under `-sm tensor`?

- **E053 cut the prize to ~3%.** The whole sparse chain is 3.2% of decode device time post-pool, so full
  4x replication would be a ~2% win, not the H9+H13-sized prize this row was written for. The counts also
  point the other way: 12 QSA layers produce 6 rtile launches per step per device, i.e. attention lands on
  2 of 4 devices as the head split implies, while the radix top-k runs 48 times per step per device
  (12 layers x 4 kernels). If anything is replicated it is selection, not attention.

- `src/llama-model.cpp:512` mirrors the indexer cache, so every device *can* run the whole chain, and
  `a8b24dfdf` shows the graph is partitioned into per-device sub-graphs with nodes landing in whichever
  split they are anchored to - the fix there was forcing a node into the first layer's split. If the
  chain's ~41 nodes per QSA layer are copied into all four sub-graphs, ~3/4 of its work is redundant and
  "run it once, broadcast the 2051 indices" beats H9 and H13 combined.
- **My earlier claim that replication is ruled out by wall clock is withdrawn**: it assumed a card here
  costs what my dev box costs, and if an R9700 is ~2x an RX 9070 then a 4x-replicated chain (~17
  ms/token at 131k) also fits the measured 15.5 ms gain.
- **Decisive without a profiler,** four arms at `-d 40960 -p 0 -n 64 -r 1` with the other flags
  unchanged: (`-sm tensor`, `-sm layer`) x (chain on, `Q4EXP_NO_INDEXER=1`). Under `-sm layer` each layer
  and its chain live on exactly one device, so that gap is the un-replicated baseline; a tensor-mode gap
  ~4x larger means replication.
- **Scope:** decode, correctness of cost model.

### H18 - why does `draft-mtp` cost a third of decode throughput on the 4-card box?

- Numbers at ctx 245760, same build either side of the clamp commit: no spec **34.86 t/s** (28.7
  ms/token), mtp after `90b9ccf9d` **23.06 t/s** (43.4 ms, acceptance 0.26), mtp before **21.5 t/s**
  (46.5 ms, acceptance 0.23).
- The arithmetic makes it a fixed-cost problem: `n_max = 6` at 0.26 accepted-per-position is ~2.5 tokens
  per step, so a step costs ~111 ms, the verify pass is ~28.7 ms of that, leaving **~13.7 ms per draft
  replay** for a 1-layer draft whose FLOPs are ~1/64 of the target's. Three candidate sources, all
  host/dispatch side:
  1. `common.cpp:1311` and `speculative.cpp:2554` force `n_rs_seq = 0` on the *draft* ctx, so any
     draft-side rewind is a checkpoint, i.e. `llama_state_seq_{get,set}_data_ext` copies per step.
  2. Every draft token is sampled and read back to host to build the next batch - 6 sync points per
     step, on a box E039/E042 already measured as gap-dominated.
  3. Under `-sm tensor` the meta dispatch cost is per sub-graph, so 6 extra replays carry the overhead of
     6 extra layers.
- **Cheapest diagnostics first:** scale `--n-draft 1,2,4,8` (linear in count = fixed per-replay,
  sub-linear = compute), then one rocprofv3 pair of MTP decode vs plain decode (E042 method) to split
  device ms from wall ms, then grep their load log for how many caches the draft ctx actually builds.
- **The same feature explains part of the VRAM pressure:** `need_n_rs_seq()` returns `draft.n_max`, so at
  `n_max = 6` the target's recurrent cache is 7x108 = **756 MiB** (their log: `size = 787.99 MiB ... 6
  rs_seq`) instead of 108 MiB, because `llama_memory-recurrent.cpp:101` allocates
  `mem_size * (1 + n_rs_seq)`. That is nearly 2x the pool's 354 MiB/card, with `common_fit_params`
  refusing to fit under tensor split.
- **And it bounds what the clamp fix buys:** qwen4exp is in `llm_arch_supports_rs_rollback`, so a
  rejection of `<= n_max` positions arrives as `seq_rm(p_keep, -1)` (clamped, good), while a larger
  rewind arrives as a `PARTIAL_ONLY` checkpoint restore that deliberately still clears the run - that
  restore leaves the indexer cells untouched, so the run's own `pos_max` still spans the rejected tokens
  and cannot bound the cut. The clear is safe by construction (the next step is `REBUILD`, which
  recomputes every row from live cells) but untested: the rollback harness exercises the `FLAGS_NONE`
  restore instead.
- Finally, the 0.23 -> 0.26 acceptance shift means the +7% is not clean evidence for the clamp; and 0.26
  acceptance on 6 drafts is a bad trade on its own terms, so `--n-draft 2` is worth a run regardless of
  the diagnosis. Full mechanics in `../../qwen4exp-arch.md` ("the two rollback doors").
- **E046 adds the ceiling:** at alpha 0.25 the most speculative decoding can do is +33% tokens per step,
  and only if the replays were free, while the measured MTP/no-spec ratio is 22.77/34.86 - so ~30 ms of
  draft-side work per step is the whole story and it is not in a kernel (rtile nb>1, `d2319c937`, was
  worth ~1.2% of it at the op level and did not register). The next check is not a measurement: whether
  `blk.N.nextn.*` exists in the GGUF at all, because `n_mtp_layers` defaults to 1 and the `n_max` clamp
  only applies under `chain_heads`, so a head-less file still drafts 6 steps. If absent, alpha 0.25 is an
  export gap (`supports_mtp_export = False` in `plans/model-shape.md`) and not a model property.
- **Superseded framing (E048):** "draft-mtp costs a third of decode" is a property of `n_draft = 6`, not
  of MTP - with 3 drafts the same box reports 37-57 t/s. At acceptance 0.25, drafts 4-6 add 0.4% to the
  accepted tokens per step, so the honest headline is "over-drafting costs a third of decode".
- **The accounting exists already; what is missing is plumbing, not a tracer:** `llama_perf_context()`
  returns per-ctx `t_p_eval_ms`/`t_eval_ms`/`n_reused` and `llama_perf_sampler()` returns `t_sample_ms`,
  both with `_reset` variants for interval deltas. `common_perf_print` (common/sampling.cpp:540-575)
  already computes `t_unacc_ms = total - (sampling + p_eval + eval)` - and the draft ctx's time lands in
  that bucket, because the function only ever receives the target ctx. Two gaps: `ctx_dft` is private to
  `common_speculative`, and `common_perf_print` is called only by `tools/completion` (llama-bench has no
  spec decode, so MTP has to be measured through the server or cli). Closing both is ~30-40 host-side
  lines with nothing ROCm-specific.
- **Settled by reading instead of measuring (was reserved as E052b):** does MTP use the QSA pool at all?
  Two halves, both answered from source.
  - The **target** side: a verify pass is an ordinary trunk forward with `ubatch.n_tokens == n_draft + 1`,
    and since `bd0b294a8` the pool covers every width below 16 by default. So yes, as the user assumed,
    and only `Q4EXP_POOLED_NO_PREFILL` at 16+ excludes anything. No run needed to learn that.
  - The **draft** side: `graph_mtp` calls the same `build_layer_attn`, which takes the QSA path iff
    `mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0` at `il = n_layer + offset`
    (`qwen4exp.cpp:989-991`). That tail entry is not the trunk's: `conversion/qwen4exp.py:88` writes
    `mtp_ratio = ratio if self._mtp_has_indexer() else 0`, keyed on the checkpoint having
    `model.layers.<mtp_bid>.self_attn.indexer.index_qk_proj.weight`.
  - So if the real file has no MTP indexer, the draft block runs **dense** attention over the whole shared
    cache: at 131k that is `131072 x 2 heads x 256 x 2 B x (K+V)` = **268 MB read per replay**, against the
    target's 2048-cell budget. One layer, but no sparsity at all. Sizing it: ~0.4 ms at ~640 GB/s, so
    ~3% of the 13.7 ms per-replay gap. Real, scaling with depth, and **not** the tax; do not promote it to
    a hypothesis on its own.
  - One command, no GPU, safe on a 111 GB file (metadata pass only): `llama-gguf <file>.gguf r 2>&1 |
    grep nextn`. Note that E046's lead is now narrower than it read: a working `draft-mtp` run already
    proves the nextn fusion weights exist, because `graph_mtp` asserts `layer.nextn.eh_proj`, `enorm`,
    `hnorm` and `hc_head_norm` non-null. What stays unknown is specifically the indexer.
- **Scope:** decode, 4-card.

---

## Next runs (mostly flag-only, bench box)

### ~~E007 - drop `-ot per_layer_token_embd=CPU`~~ killed by the user, confirmed in code

- Under `-sm tensor` the PLE table is **mirrored, not split** (`src/llama-model.cpp:513-515`), so ~30 GB
  becomes ~30 GB *per card* = ~120 GB of a 128 GB box. Not a tuning question. Replaced by
  `ple-prefetch.md`.

### ~~E008 - `GGML_CUDA_DISABLE_GRAPHS=1`~~ done: no

- Capture is active: disabling graphs costs 7% of tg and ~7% of deep pp, so the CPU split does not
  defeat it. Also bounds total launch-submission cost at ~2.7 ms/token.

### E008b - majflt/s and disk pressure during *both* pp and tg **promoted to first**

- Promoted because of H11, and widened to prefill. `pidstat -d 1` or sampled `/proc/<pid>/stat` field 12
  for **major faults/s of the llama process**, plus `iostat -x 1` (`r/s`, `%util`, `aqu-sz`), `vmstat 1`
  (`si`/`so`, to tell SSD page-cache reads from swap) and `free -h`. No build needed.
- Thousands of majflt/s in prefill against near zero in decode means the PLE table is the prefill
  bottleneck and no kernel change will show up until that is fixed. **Fold into the qsa-A redo so it
  costs nothing.**
- **User prior (2026-09-24), kept because a prior is not a measurement:** the box is "reading less" than
  earlier runs, more during prefill and less during decode. So the direction is already believed; what
  E008b still has to say is whether the prefill read rate is large enough to starve four cards, i.e.
  whether it is *the* pp bottleneck or an incidental one. The `r/s` and `%util` columns decide that, not
  the fact that reads happen.

### ~~E007b - repeat the `-sm` pp A/B after dropping `-ot`~~ deprioritised

- E007 is impossible (mirrored table) and E008 killed the capture chain, so there is no placement change
  left to re-test pp against. Source: E006.

### ~~E009 - `GGML_SCHED_DEBUG_REALLOC=1`~~ done: no

- The hook aborts when it fires and the run finished clean, so same-size realloc is ruled out. Limit: it
  only sees failed reallocs at unchanged size.

### ~~E010 - `LLAMA_GRAPH_REUSE_DISABLE=1`~~ done: reuse works

- ~22 ms/step (tg 28.20 -> 17.36; pp512 +19.6 ms per single build). Gives the magnitude class for host
  graph machinery.

### E011 - `-npl` sweep with `llama-batched-bench`

- `llama-batched-bench ... -npp 512 -ntg 128 -npl 1,2,4` (**not** `llama-bench`, which has no `-np`; in
  batched-bench the sweep flag is `-npl`, and `-np` is a separate common arg for sequences to decode).
- Cleanest discriminator, needs no debug hooks: if aggregate tg scales **super**-linearly with `npl`,
  per-step host work is being amortised over more tokens and the host path is confirmed. Sub-linear
  means it is GPU work and the host-path reasoning dies.

### E012 - `-sm row` as a third point

- Everything else as E005. `-sm` has four modes (`none,layer,row,tensor`) and E006 compared only the two
  endpoints. `row` splits weights across GPUs (parallelized) but keeps KV on the main GPU, so it
  **decomposes** tensor split into weights-split + KV-split: if row ~ tensor for tg, KV splitting is
  implicated; if row ~ layer, weight splitting is what was helping. Untested by the user.

### ~~E015 - `-nopo 1`~~ done: zero effect

- tg 28.14 vs 28.20, pp 547.5 vs 546.8; CSV confirms it applied. Scheduler op-offload is not the fixed
  cost, and this does **not** clear the lazy-CPU table - different placement path, which `-nopo` never
  touches.

### E016 - `perf record -g` during a tg-only run

- Then `perf report --stdio` as text. Flags are exhausted. This measures directly what the CPU does for
  the ~28 ms/step: allocator/graph work, QSA/PLE input construction, or blocking sync. The last bench
  measurement I would ask for.

### E014 - `-lzm off` (table resident, no demand paging) **revived**

- `[x]` Was killed on a wrong premise: it said the table is ~30-36 GB on a 32 GB box so "resident" is
  unreachable, but the box has **62.7 GiB** of RAM (H11, corrected after the screenshot), so it is
  reachable and now also feasible per E029's sizing. Also still true: `-ot per_layer_token_embd=CPU` is
  inert, because lazy ranges above 4 GiB are forced to the CPU buffer type
  (`llama-model-loader.cpp:1080-1100`). The flag that matters is `-lzm`.

### E013 - sweep `-d` at fixed `-p`/`-n`

- e.g. `-d 512,4096,16384,40960,131072`. **Justification corrected:** `-d` does fill the KV
  (`llama-bench.cpp:2408-2433`), so E005/E008 are single-depth measurements at ~40 k, not shallow ones.
  What is missing is the *curve*: depth response separates attention/KV cost from per-step fixed cost,
  and only the curve can say how much of the 35.5 ms is attention. Cheapest way to make H4b's value
  quantitative at 262 k. Source: E005.

### ~~E025 - 2-QSA-layer dummy~~ obsolete

- `--layers 8` (2 full-attn layers), or 4 layers with `full_attention_interval=2`, then tg slope vs the
  1-layer case. The prize is measured per E042+E043 (11.1 ms/token/GPU at 131k), so H9 no longer waits
  on this. Would still be nice for scaling checks, but it is no longer a prerequisite.

### E028 - memoize the QSA host mapping **(this is P2 of the H9 plan, not a flag-only run)**

- Keep `cell_blk`/`blk_cells`/`blk_pos` in the memory object, patch the `O(r)` cells a new token touches,
  let the async H2D carry the rest. Half of it is already done: `CACHED` does not create `blk_pos` at
  all, so what is left is the `blk_cells` upload (still needed by the top_k expand at
  `qwen4exp.cpp:708`) and the `O(n_kv)` scan in `try_contiguous`. In the fast state the expand is affine
  (`base_s + (sel + b_lo)*r + i`), so it can be computed in-graph from per-stream constants, and the scan
  can be made incremental against the recorded run. Still ~3.4 ms/token of host time at 131k plus ~1.2 MB
  H2D, per the E043 review.

### E029 - row-cut the real table

- The dummy's 5.5/3.7 GB variants are the precedent, so it fits in RAM, then re-measure pp t/s and GPU
  util. Causal test for H11 without touching code: if prefill util climbs toward decode's, the paging was
  the cost. Needs a conversion-side slice of `per_layer_token_embd.weight`, and it is a ~111 GB write, so
  only after E008b says it is worth it. Source: E008b.

### E030 - real text vs llama-bench's random fill, same prompt length

- Compare pp t/s, majflt/s and SSD read rate. Harness-artifact control: random token sequences hash to
  uniformly scattered n-gram rows, so bench prefill is worst-case locality and can overstate the paging
  cost relative to real serving. Source: E008b.

---

## Code-level items

### F1 - the lazy table is never prefetched, in any load mode

- `src/llama-mmap.cpp`: the `POSIX_MADV_WILLNEED` loop iterates `ranges_complement(lazy_ranges, ...)`
  (`:500-502`) - it prefetches everything *except* the lazy ranges; `MAP_POPULATE` is skipped with the
  comment "MAP_POPULATE would fault in the lazy ranges too" (`:481`); and the lazy ranges get
  `POSIX_MADV_RANDOM` (`:508-510`), which turns kernel read-ahead **off**. Also `if (numa) { prefetch =
  0; }` (`:473`) plus a whole-file `MADV_RANDOM` (`:516-522`).
- Asked and answered: `-lm none` is not the cause, since no mode warms the table. What is missing is an
  opt-in pre-warm of the lazy ranges (a background `WILLNEED` during tensor upload), or a `-lzm` variant
  for "resident but lazy-mapped". With `ple_n_heads = 16` and `MADV_RANDOM`, that is up to ~16 faults per
  token on a ~30 GB range.
- Design options, including the user's two prefetch ideas and why they split by mode: see
  `ple-prefetch.md`. `[x]` **Superseded by E031: `on-direct` removes the faults entirely rather than
  prefetching them.**

### F2 - `size_label` is cosmetic and derived from the repo/file name

- `gguf-py/gguf/metadata.py:314-328`, never computed from parameters. It reported `A3B` for a ~6
  B-active model and misled this analysis for a full round trip. Derive active params yourself;
  `llama-bench`'s `model_type` column inherits the label.
- Related consistency check that *is* trustworthy: the GGUF reports 176.94 B params vs 179.99 B in HF
  bf16, a ~3.05 B gap consistent with the exporter dropping the 1-layer MTP block
  (`conversion/qwen4exp.py:21-22`) - one MoE layer plus its attention/embedding weight.

### F3 - the PLE table is on the host because of `-lzm`, not `-ot`

- `[x]` Mechanism corrected: tensor overrides do not apply to lazy-read tensors (warning seen in E011's
  log), and lazy read forces the CPU buft itself (`src/llama-model-loader.cpp:1080-1086`), so
  `-ot per_layer_token_embd=CPU` is inert.
- The consequence stands - the table is on the host, so the gather + H2D + graph split at layer 1 are
  real - but the flag to change is `-lzm`, which is E014. E008 already showed the split does not defeat
  graph capture, and E011 bounds all per-token work at ~4-8 ms, so this is now about the ~28 ms/step
  fixed cost.
- An earlier version of this item held `-ot ...=CPU` to be the prime suspect for tg and wondered whether
  the mid-graph CPU split prevented capture; both are answered above and by E008/E011.

### T1 - implement `ggml_backend_fusion_*` for the CUDA/HIP backend

- Would unlock `test-fusion` on `ROCm0`, i.e. per-arch fusion counts and a `CUDA.csv` baseline (upstream
  ships only MTL). Only worth it if N1 (IMROPE fusion gap) or H2 need *counting* rather than timing. It
  is tooling, not a perf win - P3 died on this.
- **Demoted by E005:** counting fusions is not the bottleneck question while the box sits at 9%
  utilisation.

---

## Hygiene notes

- **`docs/ops.md` / `docs/ops/*.csv` are stale and actively wrong for these ops** `[s]`: they mark
  `DSV4_HC_*` unsupported on CUDA and Metal while the kernels exist, and `CUDA.csv` has no rows at all
  for `DSV4_HC*`/`GATED_DELTA_NET`/`LIGHTNING_INDEXER`/`TOPK`. Never cite them as support evidence.
- qwen4exp uses `llama_memory_hybrid_idx`, **not** the `dsa`/`msa`/`iswa` cache classes `[s]`.
- Sparse-ness is expressed *as the mask*; compaction is what makes a mask cheaper to iterate. No separate
  sparse kernel exists in ggml-cuda `[s]`.
- `FLASH_ATTN_EXT` has **6 known numeric failures at hsk=192/hsv=128** (gqa 8/16, permuted K/V views, err
  up to 0.0298 vs 0.0005 tol) `[v]`. Not our shape - do not chase it, but do not mistake it for a
  regression you introduced when re-running the FA suite.
- Empirical FA support at our shape: **hsk/hsv 256/256 is 142/142 supported on `ROCm0`** (f16/q4_0/q8_0
  KV) `[v]`. Zero support: 96/64, 128/64, 192/192, 64/128 - never assume a mismatched DK/DV pair works.
- This file is one heading per item on purpose. The previous table form lost a whole fragment to a
  mid-cell edit (H11, recovered from `fa67a24e8`) and silently duplicated an id (F3) when a correction
  was appended instead of replacing. Full rules: `../../record-editing-hygiene.md`.

## Explicitly out of scope for now

- Vulkan, SYCL, OpenCL, CPU-only paths. Vulkan appears above only as a reference for what a fused kernel
  looks like.
- Quality/accuracy tuning (quant selection studies, chat templates, samplers). This branch is about
  throughput and memory on RDNA4.
- Upstreaming anything. Local branch; anything that later becomes a PR gets its own discussion with the
  maintainer first, per `AGENTS.md`.

## Dead ends

### ~~E002 - a synthetic qwen4exp model as a pp/tg baseline~~ killed cheaply and on purpose

- The 19 MB F32 model fits in cache, so pp/tg measure harness overhead, not bandwidth. The point of
  keeping this: nobody should treat `tg` 325 t/s as a reference number. Replacement is E004
  (config-shaped dummy) or T2-only.

### ~~P3 - `test-fusion` counts on ROCm0~~

- The fusion debug API is Metal-only `[v]`; the tool refuses to run rather than returning empty numbers.
  See T1 if the signal is ever worth building.
