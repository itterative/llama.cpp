# Backlog, answered - kept for the record, do not re-open

Everything here is finished, killed or superseded: 32 threads. It was the bottom of
`backlog.md` until 2026-09-27, when the open material was split out so the backlog could answer "what
next" without the archaeology. Bodies are verbatim apart from the old `##` group headings.

Rules: the source of truth for a number is its `E<nnn>` record under `../runs/`, not this file. If an
item here is described as open somewhere else, that pointer is stale - fix the pointer.

- ~~B3 - dev-box ROCm version policy~~ resolved: upgraded
- B5 - rebuild the dev tree against ROCm 7.1.1
- ~~B2 - bench box facts still unknown~~ answered: ROCm 10.0.0, one Zen3 root complex, 64 GB RAM
- ~~P1 - which ops fall off the GPU~~ measured
- ~~P4 - which FA family does gfx1201 get~~ `mma_f16` for prompt processing, not for decode
- ~~P5 - does the model fit~~ capacity is not the constraint on this box
- H1 - ~~`-sm tensor` unavailable for qwen4exp forces layer split~~ revised by E005
- ~~H3 - HC kernels may be shape-restricted~~ not on HIP
- H4b - port the mask compaction to HIP, then flip `n_kv_max` **done by E020**
- H12 - put the PLE table in VRAM instead of streaming it **dead on cost/benefit**
- H13 - select QSA at block level, then expand, as the paper does **implemented**
- ~~H14 - is the model dispatch-bound at decode?~~ closed by E038: no
- H15 - keep the indexer gather and pooling in f16 **superseded by H9, ceiling cut by E053**
- ~~H17 - is the QSA selection global or per-device under `-sm tensor`?~~ no correctness bug
- H18 - why does `draft-mtp` cost a third of decode throughput on the 4-card box?
- ~~H19 - the reservation ratchet is not H9's alone~~ fixed by E069-E073, priced at +16.4% pp by E075
- ~~H23 - give mul_mat_vec_f's narrow rows more independent work per thread~~ answered by E061: real, but the allreduce absorbs it
- ~~H24 - the ~3.5 us launch/tail floor on 170+ tiny mmvf calls per step~~ mostly closed by E061: they were ILP-bound
- ~~E007 - drop `-ot per_layer_token_embd=CPU`~~ killed by the user, confirmed in code
- ~~E007b - repeat the `-sm` pp A/B after dropping `-ot`~~ deprioritised
- ~~E008 - `GGML_CUDA_DISABLE_GRAPHS=1`~~ done: no
- E008b - majflt/s and disk pressure during *both* pp and tg **promoted to first**
- ~~E009 - `GGML_SCHED_DEBUG_REALLOC=1`~~ done: no
- ~~E010 - `LLAMA_GRAPH_REUSE_DISABLE=1`~~ done: reuse works
- E013 - sweep `-d` at fixed `-p`/`-n`
- ~~E015 - `-nopo 1`~~ done: zero effect
- ~~E025 - 2-QSA-layer dummy~~ obsolete
- L1 - is the Q5 n-gram table actually resident? **closed by E058: no, and residency was never the lever**
- L2 - PLE placement is *suspected* of costing tg **closed by E058: 5.4% of the token wall, fixed to 2.3%**
- L3 - 10-of-512 expert routing on HIP **closed by E059 - routing was fine, the kernel was the problem**
- ~~E002 - a synthetic qwen4exp model as a pp/tg baseline~~ killed cheaply and on purpose
- ~~P3 - `test-fusion` counts on ROCm0~~

---

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

### ~~B2 - bench box facts still unknown~~ answered (user, 2026-09-28)

- **ROCm version: 10.0.0** (user). That is the stack B3 wants the bench box re-baselined on, and the same
  major release the dev box moved to on 2026-09-28.
- **PCIe topology:** `../results/user/lspci.log` - four Navi 48 R9700 at BDF 0b/10/13/19 under one Zen3
  root complex through two levels of PCIe switches; Navi 48 has no Infinity Fabric, so every inter-card
  byte crosses the host bridge. `../../rdna4-rocm-build.md` already carried that reading.
- **System RAM: 64 GB** (user). The 62.7 GiB the 2026-09-18 screenshot shows is the same machine after
  firmware reserve, and it is what made `-lzm off` (E014) feasible again.
- The gfx target was already known: **gfx1201 on both boxes** (user-confirmed), so a dev build is
  ISA-valid there.
- `../hw/bench-4x-r9700-32g.md` carries the three facts now; what it still lists as unconfirmed is the
  driver version, CPU model and OS.

---


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

### ~~P5 - does the model fit~~ capacity is not the constraint on this box

- See `model-shape.md`.

---


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

### ~~H3 - HC kernels may be shape-restricted~~ not on HIP

- The gate is dtype-only, all-F32, no shape restriction (`ggml-cuda.cu:5492-5501`) `[v]`. The
  `ne[1] == 4` rule I feared is Metal/Vulkan's. Replaced by N1 below.

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
- ~~**Cheapest diagnostics first:** scale `--n-draft 1,2,4,8` (linear in count = fixed per-replay,
  sub-linear = compute), then one rocprofv3 pair of MTP decode vs plain decode (E042 method) to split
  device ms from wall ms~~ **dropped 2026-09-27 on the user's call:** mtp works, acceptance runs 0.2-0.9
  by task, and `n_max = 3` is what is served - so the `n_draft = 6` tax is not on any real run's bill and
  neither the draft sweep (E048, closed) nor the E042-method trace is worth a bench session. Keep the
  load-log cache count as a free check whenever a draft log is in hand.
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

### ~~H19 - the reservation ratchet is not H9's alone~~ fixed by E069-E073, priced by E075

**Resolution (from the 2026-09-28 focus list).** SOLVED (E069-E073, 2026-09-28): fixed in `ggml-cuda.cu`, 4
lines. Cause: the HIP MoE weighted-reduction fusion (upstream #25952) reported its keep-alive allocation
dependencies only when the reduction had work, and prefill's last-layer reduction can be empty
(`ne[2] = 0`: the batch carries no output tokens). The scheduler turns each dependency into a keep-alive
view node and tests the node count before any size, so the graph's structure changed per batch, the
reservation was retightened on the first prefill ubatch, and every later ubatch - whose MoE and mask
tensors grow by one padding step - re-reserved and drained the device. `ggml_cuda_match_moe_weighted_reduction`
now takes `require_work`, false at the dependency call site and true at compute, so the fusion still skips
empty reductions but the node count no longer depends on the batch.
Measured (dev box, cli, sparse corpus, `-b 2048 -ub 1024 -c 32768`): reallocs 34 -> **0**,
`sched:realloc_size` 1689-1776 ms -> **0**, prefill wall 5539-5725 -> **4819-5075 ms**, peak VRAM
unchanged, PPL bit-identical `263100.7437`. **Dev-box llama-bench gives the fix its price tag (E075):
+16.4% pp** (`pp8192` 13248.27 -> 15419.44 t/s at `-r 3`), because pre-fix *every* allocation re-reserved
(24/24, 1218.55 ms = ~73% of the measured prefill wall) and post-fix `graph:alloc` drops from 50.9 to
0.16 ms/call.
Historical note, kept because it is the audit trail: the prize was 0.57 ms/token in decode (E062: 35
`graph:alloc` calls at 21.0 ms per 1280 decode tokens, pool on and off; E056: 13 and 120 re-reserves per
run for llama-cli and llama-perplexity) plus a larger prefill instance (E065: 34 reallocs, 1689 ms, 31%
of the wall, 1645 ms of it the forced `hsa_signal_wait_scacquire`). E066 settled the rate - one event per
256-token padding crossing, inference graph node count constant at 667 - and E067/E068 showed that a slack
on `size_max` faults the GPU while a worst-case reservation is viable but was not the lever. E069 named the
tensor (`qsa_bias`), E070 falsified the output-convention route, E071 found the single extra node,
E072 traced it to the fusion, E073 fixed it. **Invariant for any future fusion: it must not change the
graph's node count per batch.**
**T2 puts bounds on it (E074)**: the bench box's *dense* prefill shows 0 `sched:realloc*` rows in both prof
arms, so it never re-reserved and the fix is neutral for that workload - the T2 exposure is the
non-dense/pinned/vision path (H22) and the request path, and a `llama-cli`/server run with regions is what
would show it there.
Related and measured on T2 while checking this (E074): the MoE weighted-reduction fusion is worth **8.75% of
pp** (2138.37 vs 1951.32 t/s at pp4096 @ d16384).
Why the bench box is silent is **left unattributed deliberately** - H19 is closed here (E075): the flag axis
is measured and **excluded** (`--lazy-mode on-direct`, `--load-mode none`, the log's `lm` column,
`-d 16384` and the PLE-CPU override all still reallocate 24/24 or 40/40 on the pre-fix build). The two
axes that remain - the model (4-layer test model against the real one) and the 4-GPU tensor split - are not
worth a bench session on their own; the probes are recorded in E075 (on `32ba4c666` with
`GGML_ALLOC_DEBUG_REALLOC=1 LLAMA_UBATCH_DEBUG=1 -v`, or `-sm layer` against `-sm tensor`) in case H22 or
the vision path turns the same thing up.
What the closure moved out rather than archiving: the normal `graph:alloc` cost (~20.5 ms/call, ~5% of T2
prefill) went to **E016**, and the E074 prefill-utilisation observation to **H11**; the fusion's unmeasured
decode value stays noted in E074's own record.

**Original thread body (E056), verbatim:**

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

### ~~H23 - give mul_mat_vec_f's narrow rows more independent work per thread~~ answered by E061: real, but the allreduce absorbs it

- The lever works at the kernel level (`MMVF_K_UNROLL=4`: mmvf device time -21.8%, 2.42x on
  `hc_*_inject`, 1.46x on the one-row calls, 1.14x on the router) and loses at the system level:
  `ar_oneshot` +11.9%, agent 1's p90 wait 36.6 -> 140.3 us, so 76% of the saving returns as spin
  and the profiled wall ends 1.4-2% worse. The knob stays in the tree as the instrument for the
  collective work.
- **Answered on the wall:** untraced, the same two arms measure +2.04% / +1.92% tg and -1.97% /
  -1.86% on the host launch+drain total, and the traced absorption is arithmetically gone (0.475 of
  0.572 ms/token = 83% conversion; a surviving absorption would have left +0.55%). So the loss was
  the tracer, not the change. What is left is precision: E062's paired `-r 10` run, which is the
  gate for a default flip to `MMVF_K_UNROLL=4` (gated to RDNA4).

### ~~H24 - the ~3.5 us launch/tail floor on 170+ tiny mmvf calls per step~~ mostly closed by E061: they were ILP-bound

- The same calls were not launch-bound: with `MMVF_K_UNROLL=4`, `hc_*_inject` (96/step) went
  5.83 -> 2.41 us (2.42x), the `ssm` pair (72/step) 2.45 -> 1.74 (1.41x) and the one-row f32 calls
  (48/step) 2.23 -> 1.53 (1.46x). E060's block-size arms could not move them because the lever was
  never the launch path.
- So "fuse the per-layer tiny matvecs" is demoted: that 0.85 ms/step of mmvf time is now ~0.4, and
  what is left is the dispatch count itself, which E059 already parked.
- Judge it on dispatch count and device time per step, per E059's finding that 63% of kernels are
  under 2 us and the host slack tracks dispatch *count*.

---


### ~~E007 - drop `-ot per_layer_token_embd=CPU`~~ killed by the user, confirmed in code

- Under `-sm tensor` the PLE table is **mirrored, not split** (`src/llama-model.cpp:513-515`), so ~30 GB
  becomes ~30 GB *per card* = ~120 GB of a 128 GB box. Not a tuning question. Replaced by
  `ple-prefetch.md`.

### ~~E007b - repeat the `-sm` pp A/B after dropping `-ot`~~ deprioritised

- E007 is impossible (mirrored table) and E008 killed the capture chain, so there is no placement change
  left to re-test pp against. Source: E006.

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

### ~~E009 - `GGML_SCHED_DEBUG_REALLOC=1`~~ done: no

- The hook aborts when it fires and the run finished clean, so same-size realloc is ruled out. Limit: it
  only sees failed reallocs at unchanged size.

### ~~E010 - `LLAMA_GRAPH_REUSE_DISABLE=1`~~ done: reuse works

- ~22 ms/step (tg 28.20 -> 17.36; pp512 +19.6 ms per single build). Gives the magnitude class for host
  graph machinery.

### E013 - sweep `-d` at fixed `-p`/`-n`

- e.g. `-d 512,4096,16384,40960,131072`. **Justification corrected:** `-d` does fill the KV
  (`llama-bench.cpp:2408-2433`), so E005/E008 are single-depth measurements at ~40 k, not shallow ones.
  What is missing is the *curve*: depth response separates attention/KV cost from per-step fixed cost,
  and only the curve can say how much of the 35.5 ms is attention. Cheapest way to make H4b's value
  quantitative at 262 k. Source: E005.

### ~~E015 - `-nopo 1`~~ done: zero effect

- tg 28.14 vs 28.20, pp 547.5 vs 546.8; CSV confirms it applied. Scheduler op-offload is not the fixed
  cost, and this does **not** clear the lazy-CPU table - different placement path, which `-nopo` never
  touches.

### ~~E025 - 2-QSA-layer dummy~~ obsolete

- `--layers 8` (2 full-attn layers), or 4 layers with `full_attention_interval=2`, then tg slope vs the
  1-layer case. The prize is measured per E042+E043 (11.1 ms/token/GPU at 131k), so H9 no longer waits
  on this. Would still be nice for scaling checks, but it is no longer a prerequisite.

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

### L3 - 10-of-512 expert routing on HIP **closed by E059 - routing was fine, the kernel was the problem**

- **E059 closed it.** The ~115 GB/s was never a bandwidth limit: at `blocks_per_row_x = 5` only 10 of 256
  threads enter mmvq's K loop, and `rows_per_block` cannot change that because every thread already
  accumulates all of its block's rows. Sizing nwarps by K width (`d133df7d4`) cuts `mul_mat_vec_q` device
  time 1.312x and gives **+3.7..4.7% tg on the real 4-card model at every depth**. Raising rows per block,
  the obvious guess, is 4.6x *worse*. The type census is settled too: `ffn_down_exps` is **Q5_0 in some
  layers and Q8_0 in others**, not uniformly Q5_0. What survives from this row is `quantize_q8_1` (4.0% of
  device time, one launch per matvec, no memoization on `src1` at `mmvq.cu:1503-1509`) and Q8_0 at wide K
  (1.68x available at kblk=80). See [runs/E059-mmvq-narrow-k-rdna4.md](../runs/E059-mmvq-narrow-k-rdna4.md).

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

## Dead ends

### ~~E002 - a synthetic qwen4exp model as a pp/tg baseline~~ killed cheaply and on purpose

- The 19 MB F32 model fits in cache, so pp/tg measure harness overhead, not bandwidth. The point of
  keeping this: nobody should treat `tg` 325 t/s as a reference number. Replacement is E004
  (config-shaped dummy) or T2-only.

### ~~P3 - `test-fusion` counts on ROCm0~~

- The fusion debug API is Metal-only `[v]`; the tool refuses to run rather than returning empty numbers.
  See T1 if the signal is ever worth building.
