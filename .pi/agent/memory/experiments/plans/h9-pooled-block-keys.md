# H9 design - persist pooled indexer block keys at write time

Status: **built in `d8bce4e25`** behind `Q4EXP_POOLED` (default off), validated bit-identical on the
dev box. Read [runs/E044-h9-pool-implementation.md](../runs/E044-h9-pool-implementation.md) for the
six places where this design was wrong and what the validation matrix actually covers; the numbers
below are kept as written at design time.

Since then, three changes to the shape policy, all recorded elsewhere:

- **REBUILD is gone** (`8206d79d8`, [runs/E051](../runs/E051-collapse-rebuild-into-cached.md)): variant 2
  below is now CACHED with `wm = 0`, and the scores read the `set_rows` result, so there is one pooled
  topology and the write->read edge is real instead of relying on node order.
- **Widths** (`bd0b294a8`, [runs/E052](../runs/E052-pool-covers-multi-token-ubatches.md)): every width
  pools by default, `Q4EXP_POOLED_NO_PREFILL` opts wide ubatches out.
- **Reservation** ([runs/E056](../runs/E056-pooled-reserve-shape.md)): the full-cache context answers
  `qsa_pool_get` with the pooled worst case (`wm = 0`, `n_new = n_bid = ceil(n_kv/ratio)`), so
  `sched_reserve` measures the shape the runtime builds, and a run shorter than one block pools a single
  masked row instead of falling back. That removes the per-ubatch re-reservation that made pooled prefill
  cost 13-20% on 4 cards, and `Q4EXP_POOLED_NO_PREFILL` is no longer needed to protect prefill.

Source of truth for the numbers: E043 review of E042 (runs/E043-review-of-e042.md, verbatim). All
per-token-per-GPU figures use the reviewed divisor 1540 (385 tokens x 4 GPUs).

## Numbers that justify this

- Decode, 131k context, 12 QSA layers: 11.1 ms/token/GPU of the qsa arm's 31.9 is the indexer
  re-deriving all 32768 pooled block keys per token (gather 2.99 + rect copies 2.95 + rope 2.50 +
  norm 1.00 + pooling adds 1.27 + scale 0.27). 84% of the QSA chain; 35% of the arm.
- Every input to a block key is immutable once the block's 4th token is written: raw keys are stored
  once (qwen4exp.cpp:626-631), the rope position is the block START (llama-memory-hybrid-idx.cpp
  fast path fills blk_pos with b*r), the norm weight is constant. Only the top_k selection (0.55) and
  the score matvec (0.59) are query-dependent.
- Prefill pays the same per-ubatch: the chain is O(n_kv) per ubatch, O(n_kv^2) over the prefill.
  E038 measured 71 s of 430 s device time at 131k. Pooling at write time makes it O(n_kv) total.
- VRAM cost: f32 [128, n_blocks, n_stream] x 12 layers, mirrored on all 4 GPUs = 201 MB per device at
  131k; ~0.6% of a 32 GiB R9700 at the current ~28.5 GiB/card. Scales linearly with context.

## Design

### 1. Data model

- New per-layer tensor `pv` ("pooled values") in the indexer KV cache, f32,
  shape [idx_dim, n_blocks_max, n_stream], where n_blocks_max = kv_size/r, idx_dim =
  indexer_head_size (128), r = dsv4_compress_ratios[il] (4). Named `cache_idx_pool_lN` by the
  existing name-tag mechanism.
- Row b of stream s holds the FINAL block key (mean of r raw keys, scaled, rms-normed with
  index_k_norm, rope_multi at block start position) for block (b_lo + b) of that stream, exactly the
  value the score matvec reads today at qwen4exp.cpp:674.
- Bit-parity contract: per-row arithmetic is identical to today's chain because the chain is
  elementwise-per-row and pooling-at-write preserves (a) the add order ((s0+s1)+s2)+s3, (b) the
  scale-in-place f32 multiply, (c) the same rms_norm and rope_multi kernels, one row per block.
  Expected golden PPL unchanged.
- Rows that never completed a block hold garbage, exactly like today's "dead spare block"; the
  per-block bias already masks them (H13 built it; E041 records it).
- Tensor is SPLIT_AXIS_MIRRORED like cache_idx_k: add `cache_idx_pool_l\d*` to the
  pattern_idx_cache regex class in llama-model.cpp (line 393, config at :512). All four devices
  write it redundantly, same as the existing indexer chain - no new collectives.

### 2. Write path: in-graph, at raw-key write time, using existing ops only

In build_qsa_top_k, right after `cpy_k(k_raw, inp->k_idxs)` (qwen4exp.cpp:631), the cached path emits:

```
rows  = nnc_rows  // I32 [n_new]: block ids to write, host-filled
cells = ggml_get_rows(raw_cache_view, nnc_first_cells + arange(r)) // [idx_dim, r, n_new]
pool  = ((cells0+cells1)+cells2)+cells3, scale 1/r        // same order as :641-650
pool  = build_norm(pool, index_k_norm)                    // :654
pool  = ggml_rope_multi(pool, nnc_pos)                    // positions are the blk_pos rows
ggml_set_rows(view_2d(pv_s, idx_dim, n_blocks), pool, rows)   // per stream s
```

- All nodes exist today: get_rows/set_rows/cont/add/scale/norm/rope. No new kernels, no ggml core
  changes. Write-then-read aliasing inside one graph is already the pattern at :631 -> :638.
- Host supplies two tiny lists per stream: `nnc_rows` (block ids), `nnc_first_cells` (first cell of
  each block; the following r-1 cells are consecutive in the fast path), `nnc_pos` [4, n_new]. All
  three are slices of data set_input_qsa already computes (bid_idx, bid_cell, blk_pos rows).
- Decode steady state: n_new is 0 or 1 per token. To keep graph shapes constant (no can_reuse
  churn), when nothing completes the write list holds ONE duplicate entry: the most recent valid
  block (row max(wm-1, 0)), re-derived and re-written; same values, harmless. When the cache is
  empty (wm = 0) the graph variant without write ops is built (early context, rare).
- Prefill: n_new = tokens/r per stream per ubatch. Works as-is.
- Multi-stream: loop per stream over the pv views; no new machinery.

### 3. Read path

- Cached path replaces only qwen4exp.cpp:634-661: `pooled` = view_3d(pv, idx_dim, n_blocks,
  n_stream) with n_blocks = ceil(n_kv/r) from the context. The rest of the graph (q path, matvec,
  relu, head sums, bias, top_k, tail, mask, FA) is untouched.
- Fallback is literally today's chain, kept intact behind the gate - the review's "full-recompute
  fallback selected per graph build". Both variants can coexist with a shared `llm_graph_input_qsa`
  (same inputs; cached variant adds the three tiny write lists).

### 4. Freshness: generation counter + block watermark

State on llama_memory_hybrid_idx:

- `pooled_gen`: bumped by every structural mutation. Hooks, all existing methods:
  clear(), seq_rm, seq_cp, seq_keep, seq_add, seq_div, state_read() (llama-memory-hybrid-idx.cpp
  L143-251), plus llama_memory_hybrid_idx_context::apply() when it runs with no ubatches - that is
  the KV-cache update() path where defrag/stream copies move cells physically (L810-818; no
  upstream changes needed, the wrapper sees it).
- `pooled_wm`: first block id not yet pooled; rows [0, wm) valid for the current gen. Reset to 0 on
  any gen bump. Advanced in llama_memory_hybrid_idx_context::apply() by the n_new of the graph that
  just executed (apply() asserts status success, so an early exit cannot over-advance).

Three graph variants, chosen at build time, honest to can_reuse:

1. cached-read + incremental write: steady decode/prefill after the cache is warm. The work per
   token collapses to the required set: matvec 0.59 + top_k 0.55 + mask/compaction 0.67 + FA 0.07 +
   head adds + a one-block write (~launch overheads). Expected ~2.3 ms vs 13.2 today.
2. rebuild-read + write-back: first graph after a gen bump. Runs today's full chain AND writes all
   n_bid rows back via one set_rows. Cost = today's 11.1 ms + ~1 ms, ONCE per invalidation.
3. recompute-read, no cache: the general/multi-seq path (or Q4EXP_POOLED unset) - today's graph,
   no cache involvement at all. The cached path is only ever enabled for the contiguous
   single-sequence fast state, because the general path's block ids are renumbered per ubatch and
   are not append-stable (llama-memory-hybrid-idx.cpp group machinery).

This keeps the risky surface to "did we notice the mutation": a missed hook leaves stale rows, but
stale rows are always either (a) masked by the block bias, or (b) caught by the gen check that
switches to variant 2. Worst case of a missed hook is wrong selection, detectable by the
selection-overlap test below.

### 5. Precision

- v1: f32 (user decision, agreed). Bit-comparable to today; no re-baseline needed beyond confirming
  golden PPL.
- f16 later as an optional knob: halves VRAM and the matvec read but rounds the cached key, so the
  selection can shift last-bit; needs its own validation pass.

### 6. State save/load and the RAM prompt cache

- v1: pooled values are NOT persisted. state_read bumps gen -> the first graph after load is a cheap
  rebuild (11.1 ms once). No state format change.
- Checked: `-ln/--cache-ram` ("cram", PR 16391) is not a separate mechanism - `prompt_save` /
  `prompt_load` in tools/server/server-context.cpp call `llama_state_seq_get_data_ext` /
  `llama_state_seq_set_data_ext`, i.e. exactly the per-seq `state_write`/`state_read` path above
  (llama-memory-hybrid-idx.cpp:208-251). So cram restore is covered by the same gen hook, and
  skipping pv in v1 also keeps the saved blob byte-identical to today.
- Persisting pv later would grow the blob 4x for the indexer part and needs a format version bump -
  deferred, not complex, but no measured benefit yet (one 11 ms rebuild per restore).

### 6b. Verified non-issues (checked while designing, so they do not have to be revisited)

- `ctx_other`: wired only for GEMMA4_ASSISTANT, EAGLE3, DFLASH (llama-context.cpp:145-162) and it
  feeds `mem_other` for shared-cell draft caches. The indexer cache is constructed with
  `mem_other = nullptr` (llama-memory-hybrid-idx.cpp:19-68), so nothing here interacts with it.
- MTP: `hparams.n_layer_nextn` is not set for qwen4exp (no nextn tensors), and LLM_ARCH_QWEN4EXP is
  not in the `mtp_on_hybrid_qwen` list (llama-model.cpp:2582-2589), so there is no draft/MTP context
  over this cache. Rewind-style reuse (re-decoding over existing positions) re-writes the raw key
  with a deterministic-equal value and the pooled write re-derives the same row, so it is
  self-consistent and needs no special handling.
- Write timing must stay at completion, not the next token: today's completing token already scores
  its own block (the ubatch's cells and positions exist when set_input_qsa runs, which is what
  `tail_cells` and the causal binary search in the same function rely on). Deferring the write one
  token would make that row unwritten at read time and change the selection. The in-graph write
  after `cpy_k` and before the matvec reproduces the current semantics exactly.

### 7. What this is NOT (from E043, kept for scope discipline)

- Does not touch the mask build, the compaction (F7b grid fix is independent, ~0.57 ms ceiling, do
  it anyway), NCCL, or the mirrored-split arrangement.
- Does not fuse norm+rope into the matvec (the bespoke-kernel recommendation 2) - H9 first; the
  kernel later runs 1 new block per 4 tokens instead of 32768.
- Does not fix F8 (FA half-count ambiguity) - re-measure separately.
- Env gate `Q4EXP_POOLED` (default OFF until parity and A/B bench land, then ON), same style as
  Q4EXP_SPARSE_FA / Q4EXP_CELL_SEL.

## Implementation order

- P1 (this design): cache tensor + watermark/gen + write/read graph variants + gate. ~250 lines.
- P2 (E028, review's bonus): stop uploading blk_cells/blk_pos for all blocks - the host scan is
  ~3.4 ms at 131k plus ~1.2 MB H2D per token. With the cache, blk_pos is needed only for the new
  block(s) and blk_cells only for the top_k expand; in the fast path the expand is affine
  (base_s + (sel + b_lo)*r + i) and computable in-graph from per-stream constants. Do after P1 is
  proven; it removes the last O(n_kv) per-token host work.
- P3: f16 pooled cache knob (optional).

Files touched by P1:

- src/llama-kv-cache.h/.cpp: kv_layer gains `pv` (nullable); allocate in ctor when
  model.arch == QWEN4EXP and hparams.indexer_head_size > 0; accessor + size accounting + log line.
  State write/read leaves pv out automatically (separate field, not k/v).
- src/llama-model.cpp: extend pattern_idx_cache to match cache_idx_pool_lN (mirrored placement).
- src/llama-memory-hybrid-idx.h/.cpp: gen + wm state, hooks in clear/seq_*/state_read/context
  apply, extended set_input_qsa signature computing nnc_rows/nnc_first_cells/nnc_pos for rows
  [wm, n_bid), plus a "one duplicate write" rule for the steady decode case.
- src/models/qwen4exp.cpp: input fields + set_input + can_reuse extension, cached read branch, write
  chain branch, fallback kept whole, Q4EXP_POOLED gate.
- No new files. No tests/* additions (repo rule): validation reuses existing infra.

## Validation

- Golden PPL: must stay 263113.6984 (Q4EXP_POOLED=1 vs 0, same build, 131k dummies).
- Selection-differential: E037's host-dump technique or an overlap-count print - the top_k sets of
  recompute vs cached builds must be identical (not just the PPL).
- llama-bench A/B: tg128@131k, qsa arm, PLAIN output. Expect ~31.9 -> ~21-22 ms/token/GPU and
  prefill 430 s -> ~370 s device (71 s chain => ~11 ms/4 erased, whatever E038's split says).
- Sequence-op stress, PLAIN normal usage: a scripted session with seq_add/seq_div (K-shift),
  seq_rm/cp, state save/load mid-generation - verify outputs match between Q4EXP_POOLED=1 and 0
  and that wm/gen transitions produce clean rebuilds. This is the correctness gate for section 4.
- Multi-GPU: same-dummy pp runs on the bench box; mirrored writes are redundant per device, so
  traces should show the write ops on all 4 agents like the rest of the chain.

## Signs we are wrong / risks

- Cross-graph write->read ordering: relies on the same guarantee the raw cache already uses
  (cpy_k in graph N read by get_rows in graph N+1). If a backend regressed here it would break
  today's path too.
- The watermark/gen discipline is the cost. Mitigation: gen hook surface is the wrapper's own
  seq_*/clear/state_read (all local) + one apply() branch; section 4 semantics make a missed hook
  noisy (selection diff) rather than silently wrong-but-plausible.
- Added f32 traffic for cached-path + duplicate writes is negligible (r=4 rows at decode).
- If P1 lands with no prefill gain in practice, stop and re-trace before doing P2 - do not stack
  unverified savings.

## What needs sign-off before code

1. Overall shape: mirrored [idx_dim, n_blocks, n_stream] f32 cache in the existing idx KV cache object.
2. Write-at-write-time in-graph with existing ops, duplicate-write trick for constant graph shapes.
3. Rebuild-on-invalid path (variant 2), fallback = today's chain (variant 3), cached path gated to
   the contiguous single-seq fast state.
4. Pooled values not persisted in state files for v1; rebuild on load.
5. Gate Q4EXP_POOLED default OFF -> ON after parity + bench.