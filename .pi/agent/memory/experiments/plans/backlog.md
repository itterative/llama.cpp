# Backlog - candidate threads, qwen4exp on RDNA4

Not experiments yet. A thread becomes an `E<nnn>` id only when it has a falsifiable hypothesis and a
deciding metric (see `../PROTOCOL.md` 3.1).

Anchors at commit `ebbb18522`. Markers: **[v]** read directly by me, **[s]** from the `scout-1` survey
and not re-read, **[x]** corrected after verification. Backend picture lives in
`../../rdna4-rocm-build.md`; real dims in [model-shape.md](model-shape.md).

Format: one item per `###` heading, `id - question`, with the fields as bolded labels underneath. A
struck-through id means the thread is answered or dead; the answer stays inline, because these notes
are why later experiments were scoped the way they were.

**Live right now:** H20/H21/H22 (what E057's fix left open: the pool never engages for a session with
an image, `qsa_pool_get` promises more than the fast path verifies, and whether the vision path pays
H19's ratchet), H19 (the reservation ratchet outside H9: llama-cli and llama-perplexity re-reserve
with the pool off), H18 (the MTP tax, now framed as over-drafting), H17b (is the chain 4x replicated),
E008b (the measurement that decides the whole PLE/prefetch line), and the H13 leftovers. H9's prefill
question is closed by E056, H9's correctness under mrope by E057.

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

### L1 - is the Q5 n-gram table actually resident? **closed by E058: no, and residency was never the lever**

- Answered by counting rather than guessing: `/proc/self/io` deltas inside `gather()`, reported through
  `ggml_prof_count` (`867f3eed3`, `126b7a43b`) and bucketed decode vs prefill by row count. `rchar` is
  every byte asked of `pread` including cache hits, `read_bytes` is only what storage served, so the pair
  is the miss rate with no external sampler and no model-load contamination.
- **The stored row is 110 B** (one Q3_K block of 256 elems), not the 1760 B that a Q5 reading of
  `ple_embed_dim = 2560` implies. 20M n-gram vocab x 16 heads = 320M rows x 110 B = 35.2 GB = 32.8 GiB,
  which is H12's size. A decode step therefore reads **16 distinct rows** (`io:uniq_decode` ==
  `io:rows_decode` in eight runs), and a 1024-token prefill ubatch reads **12,114**, not 756.
- Prefill is ~99.9% warm whenever the cache holds (42.93 kB of storage for 12,114 rows). Decode is not:
  26-33 kB/call = 6.5-8.2 cold pages out of 16 rows. Different populations, so what prefill warms does
  nothing for decode. The user's "1-2 MB/s during decode" was MB/s and it reproduces: 112.87 MB over 74.6 s.
- **Residency is not the lever, because the misses are compulsory first touches, not evictions.** 2820
  tokens introduced 33.2k distinct rows = 133 MB of pages, trivial on a 62.7 GiB box. `io:reuse_decode`
  is 27.7-34.1% and those rows are already free via the page cache, which is why storage is 26-33 kB/call
  and not the 64 kB/call of 16 cold pages. That closes **every row cache, host or GPU**: a cache can only
  re-capture what the kernel already captures, and a page-cache hit is ~1-2 us.
- **Retracted:** the `-lzm off` A/B. With no reader the gather becomes a `ggml_get_rows` CPU op inside
  the graph again, so that arm measures the split E031 removed rather than residency.
- **Retracted:** `POSIX_FADV_RANDOM` (`b9301a5f1`, dropped). 9.8 pages for 16 rows means at most one page
  per row, so there is no readahead amplification to remove; the arm that appeared to show one had a cold
  cache (prefill 73% cold, `storage/rchar` 0.032x -> 27x on byte-identical requests).
- **The lever was concurrency** - see L2.

### L2 - PLE placement is *suspected* of costing tg **closed by E058: 5.4% of the token wall, fixed to 2.3%**

- Stale as written: under `-lzm on-direct` there is no `ggml_get_rows` node at all.
  `llm_graph_lazy_rows::build` returns an F32 input tensor, the host pre-gathers and dequantizes, and
  `set_rows` uploads it, so the fetch moved out of the graph and into `graph:set_inputs`. E031 had already
  taken the mid-graph split away, which is why no cache proposal could claim that prize.
- **The cost and its mechanism:** `input:lazy_gather` was 1.4181 ms of a 26.32 ms token (5.4%), exposed
  because `set_inputs` runs before the enqueue and after the previous step's sync. 16 independent rows
  read by `n_workers = min(n_readers, max(1, n/32))` = **1** at n=16, so the ~8 cold ones were 8 serial
  queue-depth-1 waits at 174 us each.
- **Fixed and landed as the default** (`3f1138bb3`): `POSIX_FADV_WILLNEED` for every distinct row before
  waiting on any. Gather 1.4181 -> **0.5819 ms/call**, 174 -> 78 us per cold page, tg **38.0 -> 39.6
  (+4.2%)**, prefill unchanged (1292.0 -> 1290.1 t/s), storage unchanged (33.4 -> 30.5 kB/call). Each
  arm's tg gain matched its own gather delta to within 0.3 t/s, so the attribution does not need reps.
  `LLAMA_LAZY_PREFETCH=0` opts out; `LLAMA_LAZY_WORKERS` stays for tuning.
- Do not quote `gather / phase:decode`: `phase:decode` (`llama-context.cpp:1734`) wraps only
  `llama_decode`, i.e. the host side of a step (7.48 ms), and the other ~18.6 ms/token is the sync, the
  logits read and sampling. That ratio overstates the share 3.5x.
- **Dead, as measured:** the worker divisor as a default (it works, 174 -> 109 us/page, but pays a
  0.29 ms/token thread-spawn floor visible in `gather min`), `lazy_staging` (27-30 ms total, its 4.4 ms
  max being the one-time zero-fill of a 16.8 MB grow at `-ub 1024`), `lazy_h2d` (19-24 ms total, and the
  set is async so that is enqueue time) and `lazy:sort` (20 ms total).
- **Still open, the tail:** `lazy_gather` max is 28.97-30.63 ms in all four A/B arms and 51.77-123.62 ms
  in runs 3-4, so one row read can cost more than a whole token's budget. Prefetch cut the mean 59% and
  barely moved the max, so this is not queue-depth latency. Irregular tg will still show it.
- **Still open, warm-cache cost:** with rows already cached the prefetch is pure overhead - gather 0.1194
  -> 0.1687 ms/call on the dev box, ~0.5% of a bench-box token against the +4.2%. A size threshold would
  fix that and needs a magic number to defend.
- **Still open, and bigger than PLE:** `meta:subgraph` is 97 dispatches and ~5.4 ms per decode step,
  essentially all of decode's `graph:compute` and ~21% of the token wall, now **3.4x the fixed gather**,
  with `meta:allreduce` adding ~1.4 ms/step. Comms thread.
- **Method, worth remembering:** `--temp 0` makes this model loop, and `-s <seed>` did not reproduce
  across runs with `-sm tensor` (probably 4-card reduction order moving the last logit bits). So no
  llama-cli A/B on this box can be text-matched; normalize per call, and use the deterministic part of the
  workload as a control - prefill's counters came out byte-identical across all four arms.

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

### H9 - cache pooled indexer block keys (coarse cache) **shipped; prefill no longer excluded (E056)**

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
- **Resolved (E056):** the reservation now measures the pooled shape. `qsa_pool_get` answers the pooled
  worst case (`wm = 0`, `n_new = n_bid = ceil(n_kv/r)`) when the context is a full-cache one - keyed on
  `is_update`, the same flag that already fakes `ns_ubatch` for reservation, not on the cell state, which
  is what made E051's attempt fail - and a run shorter than one block pools a single masked row instead of
  falling back to the historic chain, because the tools' 1-token warmup was enough to drop the pooled
  budget and restart the ratchet. Result on the dev box, sparse FA + rtile, 3 interleaved passes:
  **0 re-reserves** in every pooled arm (was 24+7), reservation +0.08 MiB, pp8192 702.43 vs 702.08
  pool-off, tg128 35.86 vs 34.38, all nine numeric gates bit-identical. So the E052 `-2.9%` was the
  reservation after all, and what it cost on 4 cards is a fixed ~110 ms per prefill ubatch of lost
  host/device overlap (run8: pp8192 1781 vs 2230), not device work. `Q4EXP_POOLED_NO_PREFILL` is now
  optional.
- **Answered on 4 cards (E056's T2 run):** pooled prefill at d131072 gives pp8192 1454.65/1463.35 against
  1436.33/1428.98 with prefill excluded and 1423.78/1425.67 pool-off, i.e. **+1.8% / +2.4%**. Under the
  protocol's 5% bar, but cleanly ordered across 6 alternating samples, and the mechanism shows up in the
  host numbers rather than the device ones: the pooled variant never creates `blk_pos` (I32
  `[4*n_blocks*n_stream]`, 557 KB per ubatch at 131k), so `graph:set_inputs` runs 90.8 ms/call against
  125.9. That is E028/P2's prize arriving for free. The reservation is also **34.8 MiB smaller**
  (2319.04 vs 2353.85 MiB), and tg is +32.6% over pool-off, matching E045. So
  `Q4EXP_POOLED_NO_PREFILL` has no job left.
- **Default flipped on in `9111adf2c`**, and `Q4EXP_POOLED_NO_PREFILL` is deleted with it: parity plus a
  bench number were the two conditions the plan set, and E056's T2 run met both. Verified after the flip,
  on the dev box: the sparse corpus at `-c 8192 -b 2048` and the `-b 256 -c 2048` gate both reproduce
  their recorded pool-on values (267035.3875, 266571.9557) with no env set, the cache logs
  `block key pool = 1` and 49 `qsa pool: mode = 1` lines where E056 saw 49, and `Q4EXP_POOLED=0` still
  gives `block key pool = 0` with zero mode lines and the same PPL. The other seven gates were not re-run:
  E056 already diffed all nine pool-off vs pool-on on the same build, and a default flip cannot change
  what either side computes.
- **The tradeoff the flip accepts:** E050 measured the pool as *negative* for tg below ~32k (-3.7% at
  d4096, ~0 at d16384, +2.6% at d40960, +30.1% at d131072), so short-context decode pays a little for a
  feature it does not use. Those numbers predate the reservation fix and the flip, and E013's `-d` sweep
  is the run that would re-draw the crossover. The branch targets 262k, which is why this is acceptable.
- **Still open:** states that are not one dense sequence (multi-seq under `--kv-unified`, an interior
  `seq_rm`) build the historic graph and so churn against a pooled reservation; closing that means making
  the general path build the pooled topology too (~40 lines in `set_input_qsa`, which already computes
  the per-block cells and positions).
- Also open, and now on every run's bill: the pool taxes **354 MiB/card at ctx 245760**, so f16 storage
  (H15/P3) went from "on the table" to "the default's cost"; and the crossover depth where pooling stops
  paying is unexplained.
- **Separate bug found on the way (E056):** `llama-cli` and `llama-perplexity` re-reserve 13 and 120
  times per run **with `Q4EXP_POOLED=0`**, same ratchet, different first trigger. Nothing to do with H9;
  promoted to **H19** with the named tensor and the evidence.
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
- Still open on this item: the 256 KiB size cutoff is a guess, and whether `GGML_CUDA_ALLREDUCE` should stop
  defaulting to NCCL on this branch. Both env-var plumbing questions are closed: `auto` reaches one-shot
  since `4cf0555a4` (two variables to opt in), and the inboxes are sized off the cutoff rather than
  `GGML_CUDA_AR_DIRECT_TMP_BYTES` since `git log -1`, so they cost 4 MiB per GPU instead of 256 MiB.
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

### H19 - the reservation ratchet is not H9's alone: llama-cli and llama-perplexity re-reserve with the pool off

- **Found by E056**, which removed the pool's own contribution and then measured the residue. Counts on
  `q4exp-4l`, `-sm none`, **identical with `Q4EXP_POOLED=0` and `=1`**: `llama-cli -c 4096 -n 2500 -st`
  13 in both arms, the `-c 1024 -n 2000` shift stress 6 in both, `llama-perplexity -b 256 -c 2048` 120 in
  both (its runtime node count alternates 696/697, 30 of the 120 on the count). `llama-bench` pp/tg: 0.
  Note `llama-cli` in this build runs on the merged server machinery, so its log carries `srv`/`slot`/`que`
  lines.
- **Same three-act ratchet as H9's.** The reservation is `n_tokens 512, n_seqs 1, n_outputs 1` -> 697
  nodes / 138 leafs, 231.39 MiB. Then, in order: a node-count change at t=2.263 s, `node
  model.input_embed is not valid` (a size) at 2.343, a second node-count change at 2.365 - all inside the
  first 110 ms of the prompt phase - and after that ten size events spaced exactly ~1.06 s apart.
- **The 1.06 s is 256 generated tokens, and 256 is the cache's own padding:** `llama_kv_cache::get_n_kv`
  (`llama-kv-cache.cpp:1260`) rounds n_kv up to `max(n_pad, 256u)`, "so that the graph remains constant
  across batches and can be reused". So `can_reuse` holds for 256 decode steps (no alloc at all), then
  n_kv jumps and every depth-proportional tensor jumps with it.
- **The growing tensor is named.** `leaf_111` is the src of `node #517 (GET_ROWS)` whose other src is
  `cache_idx_k_l3`, i.e. `ggml_get_rows(k_all, inp->blk_cells)` in `build_qsa_top_k`, so it is
  **`blk_cells`**, I32 `[ratio*n_blocks, n_stream]`: 16 KB at the reserve's n_kv=4096 (4 B x 4 x 1024
  blocks) and stepping 1K -> 2K -> 3K in the `GGML_SCHED_DEBUG=2` assignment listing. `attn_inp_kq_mask`
  and the QSA block bias grow alongside it.
- **The ten decode events are collateral, not the bug.** n_kv can never exceed n_ctx, so had the
  worst-case budget survived the prompt phase there would be zero re-reserves in the whole session -
  which is what llama-bench shows, and what E056 bought for the pool.
- **Gap: the root cause is unnamed.** What are the two node-count changes? The reserve is built with
  `n_outputs = 1` and `n_seqs = n_seq_max` and passes `sampling.samplers` into
  `ubatch_prepare_reserve`/`resolve_fused_ops`, while the server's prompt ubatches carry 0 outputs except
  the last one, so the output/fused-sampler path is the only part of the graph whose node set can depend
  on that. `build_inp_out_ids` deliberately keeps its topology constant (its comment cites PR 14275), so
  it is not the obvious candidate; perplexity's 696/697 is a single node, which fits that family.
  **Cheapest probe:** three lines in `process_ubatch` printing `ggml_graph_n_nodes(gf)`/`n_leafs` next to
  n_tokens/n_outputs, or re-add E056's split-graph dump. One 90 s run.
- **The user's read is that this belongs to the utils, not to `llama-server`.** The evidence so far points
  the other way (cli and perplexity churn, bench does not), but it does not settle it: `llama-cli -st` is
  one slot with no prompt-cache reuse and no keep-alive, so a real server session is untested. One curl
  request against a server started with `GGML_PROF_REGIONS=1` and a `sched:realloc` count decides it.
- **Cost:** ~25 ms per event here, ~300 ms on 4 cards (run8's `realloc_size` ms/call). A 2500-token turn
  throws away ~0.3 s locally and ~4 s on the bench box, and a 4000-token generation re-reserves ~16 times.
  It is a per-turn tax rather than a per-prefill one, which is why it hides under everything else.
- **Scope:** every tool, both split modes, and probably not qwen4exp-specific once the node-count
  difference is named - any arch whose host inputs scale with n_kv ratchets the same way, it only takes
  one early mismatch to start it.

### H20 - give vision sessions the pool: pool block keys in rank/cell space, not position space

- **Found by E057.** `qsa_pool_get` requires the used run to be dense in position
  (`llama-memory-hybrid-idx.cpp:379-388`), and an mrope image pins `nx*ny` cells to one position, so
  **the pool never engages for any session that contains an image**, in either context, and only
  re-engages once those cells leave the run. E044..E056's `+32.6% tg` therefore does not apply to
  vision chat at all, and nothing in the gate suite could show it: every existing gate feeds text
  whose positions step once per cell.
- **Why rank space is poolable.** Rank order equals append order for mrope (image cells share `t`
  and sort by `(y, x)`, the next text token's `t` is above the whole image), so a rank-space block
  number is append-stable exactly like a position-space one, and E057 made the general path read and
  write in rank space already. What has to change is the pool's own keying (`b_lo`, `wm`, the
  `qsa_run` record) plus whatever assumes `blk_cells` columns are position buckets.
- **Cost if not done:** a holed or pinned run also loses the pooled reservation. E057 measured 19-21
  `sched:realloc_size` at ~120 ms per 7.5k-cell run against 0 for dense, which is H19's ratchet (and
  the count disagrees with wall time there, see H22).

### H21 - `qsa_pool_get` promises on an endpoint test that `set_input_qsa` does not honour

- **Found by E057.** The pool decision is made from endpoints (`p1 == p0 + (j1-1-j0)`), the fast
  path additionally verifies every cell. A run with both duplicates and jumps can satisfy the first
  and fail the second, and then the build-time promise is `CACHED` while the runtime falls back to
  the general path. E057 made that state safe (row/cell 0 are named before the erase) but it costs a
  graph rebuild per 256-cell bucket, because each rebuild answers with a new `n_new`.
- **Fix shape:** carry a "verified through" cell index in the `qsa_run` record so the promise and
  the verify test the same thing incrementally, instead of making `qsa_pool_get` walk all cells (it
  also runs from `can_reuse`, i.e. once per ubatch).

### H22 - is the vision path really paying ~120 ms per re-reserve? (bench reading, blocks on nothing)

- **Open question from E057**, which could not settle it: on the dev box a `pin512` or `gap512` run
  at 7.5k cells counted 19-21 `sched:realloc_size` rows totalling ~2.5 s, while the wall time moved
  by 0.2 s (16.76 s dense vs 16.96 s pinned). Either the region accounting is inflated (PROTOCOL 5
  warns: inclusive totals, unlocked counters) or the cost hides behind the device queue.
- **Why it matters:** on 4 cards a re-reserve measured ~300 ms, so the same 20 events per 7.5k cells
  would be ~6 s per turn of vision chat. Decide with `GGML_PROF_REGIONS=1` plus wall time on one
  real 3-image conversation, not with more local reps.

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
