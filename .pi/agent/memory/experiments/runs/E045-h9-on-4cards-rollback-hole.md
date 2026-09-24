# E045 - H9 on the 4-card box: tg +30% at 131k, pp -16..-22%, and a rollback hole in the design

> Bench box (4x R9700), real checkpoint, build `36ec37826` with `Q4EXP_POOLED=1`, against the
> pre-H9 baseline build `93136fa41`. Same flags both runs: `-lm none -sm tensor -fa 1 -lzm on-direct
> -ot per_layer_token_embd=CPU -r 3 -b 2048 -ub 1024`, plus `GGML_FATTN_RDNA_RTILE=1
> Q4EXP_SPARSE_FA=1`. Raw: [results/user/results-post-h9.log](results/user/results-post-h9.log).

## The result

| test | pre-H9 | pool on | delta |
| --- | --- | --- | --- |
| tg128 @ d131072 | 23.16 +/- 0.58 | **30.14 +/- 0.93** | **+30.1%** |
| tg128 @ d40960 | 30.29 +/- 1.03 | 31.84 +/- 1.05 | +5.1% |
| tg128 @ d16384 | 32.36 +/- 1.21 | 32.28 +/- 1.07 | ~0 |
| tg128 @ d4096 | 33.65 +/- 1.27 | 32.39 +/- 1.12 | -3.7% |
| pp8192 @ d131072 | 1502.33 +/- 1.23 | 1260.05 +/- 4.39 | **-16.1%** |
| pp8192 @ d40960 | 1955.58 +/- 1.20 | 1594.10 +/- 0.75 | **-18.5%** |
| pp8192 @ d16384 | 2146.31 +/- 0.90 | 1713.25 +/- 1.87 | **-20.2%** |
| pp8192 @ d4096 | 2262.04 +/- 1.70 | 1766.27 +/- 1.63 | **-21.9%** |
| pp4096 @ d131072 | 1464.60 +/- 5.95 | 1250.89 +/- 4.57 | -14.6% |
| pp512 @ d131072 | 1009.99 +/- 89.22 | 967.75 +/- 58.74 | -4.2% |

Decode at the depth where the chain was measured (E042/E043) moved as predicted: 23.16 -> 30.14 t/s
is -8.9 ms/token against a predicted -11 ms. The shallow depths are flat or slightly negative, which
is also expected - at 4096 the pool only covers 1024 blocks.

**Prefill regressed 16-22% at every depth, including d4096 where the chain is small.** Unexplained
so far. The dev box shows -3.3% pp on the same build pair, so whatever it is needs >1 device (the
meta backend is in play under `-sm tensor` even on 1 device, but only with 1 backend there). The
pending discriminator is pool=0 on the same build (isolates pool vs the meta fix + roctx patch, the
only three commits in between) and a `-ub` sweep, which separates per-ubatch overhead from per-token
work. GPU utilization dropped to 50-70% on both pp and tg after the change, which points at host or
transfer stalls rather than device work.

## The hang, and the design hole it exposed

The server (same build, `Q4EXP_POOLED=1`) took a GPU hang: `HW Exception by GPU node-2 ... GPU Hang`,
`MES failed to respond to msg=REMOVE_QUEUE`, MODE1 reset, DRM devcoredump. Artifacts under
[results/user/lcpp_coredump/](results/user/lcpp_coredump/). Config of interest: `ctx-size = 245760`,
`spec-type = draft-mtp` with `n_max=6`, `ctx-checkpoints = 32`, `verbosity = 4`.

- **The pool costs 354 MiB per card at that ctx**, straight from their own log: the indexer cache
  reports `KV buffer size = 1080.00 MiB` against `K (f16): 720.00 MiB`. Free VRAM went 130216 ->
  22832 MiB aggregate during load, and `common_fit_params` refuses to fit under tensor split
  (`llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`), so nothing backs the context
  off. One card was observed at 32/32 GB. Not proven as the hang cause, but it is a plausible last
  straw, and it is the first time the pool's memory has actually mattered.
- Log timestamp prefixes are `MM.SS.mmm.uuu`, not days or hours: the process was ~16 minutes old, not
  15 hours. My first read was wrong and the user's was right.
- Ruled out by reading: the pool is not being split across devices. `llama_meta_device_get_split_state`
  in `src/llama-model.cpp` is the only name matcher in the tree (`cache_idx` appears nowhere else),
  and its regex `cache_idx_(k|v|pool)_l\d*` matches `cache_idx_pool_l0` -> `SPLIT_AXIS_MIRRORED`. A
  split pool would have meant `set_rows` row ids past each device's shard, i.e. a wild VRAM write, so
  this was worth checking first.

**The real hole:** I had assumed speculative rollbacks arrive as a partial `seq_rm`. They cannot:
`llama_memory_hybrid_idx::seq_rm` asks the recurrent cache first and GDN refuses a partial removal,
which is exactly why llama.cpp has context checkpoints for hybrid models. Checkpoints are
`llama_state_seq_get_data_ext` / `set_data_ext` (common.cpp `common_prompt_checkpoint::update_tgt`),
i.e. the per-sequence `state_read` path - and that hook did `qsa_pool_invalidate()`. With `n_max=6`,
most decode steps reject something, so the pool would re-derive every complete block, all 61440 rows x
12 layers, on nearly every step: strictly worse than the historic path, and a good candidate for the
low utilization.

## The fix - clamp instead of clear on a truncation (`90b9ccf9d`)

A cut at position p0 leaves every block ending below it intact, so the watermark clamps to
`n_keep = (p0 - r)/r - b_lo + 1` and `j1` follows the surviving cells; only rows above it are
re-derived as they complete again. Applied to per-sequence `state_read` (the checkpoint/rollback path)
and to a `seq_rm` ending at -1. Interior removals, `seq_add`/`seq_div`/`seq_cp`/`seq_keep`, `clear`,
`state_drop` and whole-context loads keep the wholesale clear.

Validation, including a rollback test the earlier matrix lacked
([tools/qsa-rollback-harness.cpp](../tools/qsa-rollback-harness.cpp), throwaway, extracts a
checkpoint, continues, re-inserts and replays):

- pool off vs pool on: identical token streams (`cksum=4903597782947487587` both arms) and
  `rollback_replay_mismatch=0` - restoring a checkpoint reproduces the continuation exactly.
- 1 rebuild across two restores with 350 cached steps (before the clamp: a rebuild per restore).
- golden PPL 263113.6984 both arms; `test-save-load-state` all pass with the pool on; sparse corpus
  `-sm tensor -c 8192` gives 267035.1801 in both arms (the `-sm none` value 267035.3875 differs by
  ~1e-6 because tensor splitting changes reduction order, and both arms move the same way).

## Addendum - the clamp on the real MTP path (user's numbers, same day)

At ctx 245760 on the 4-card box: no spec **34.86 t/s**, mtp before the clamp **21.5 t/s** (acceptance
0.23), mtp after **23.06 t/s** (0.26). So MTP still costs a third of decode throughput, and the gain
is small - with the acceptance rate moving between the two runs, it is not clean evidence that the
clamp did anything. Two things came out of reading the path properly:

- qwen4exp is listed in `llm_arch_supports_rs_rollback`, and `need_n_rs_seq()` returns `draft.n_max`,
  so the target ctx gets 6 recurrent snapshots and a rejection of <= 6 positions lands on
  `llama_memory_hybrid_idx::seq_rm(p_keep, -1)` - the branch the clamp was written for. A *larger*
  rewind lands on `load_tgt(PARTIAL_ONLY)`, which still clears: that restore does not touch the
  indexer cells, so the run's own `pos_max` still spans the rejected tokens and cannot bound the cut.
  Clearing there is safe (the next step is `REBUILD`, recomputing every row from live cells) but it is
  also untested - the harness restores with `FLAGS_NONE`, i.e. the whole-state path.
- The same knob costs VRAM that nobody priced: `n_rs_seq = 6` puts the target's recurrent cache at
  **756 MiB** instead of 108 (their load log: `size = 787.99 MiB ... 6 rs_seq`), because
  `llama_memory-recurrent.cpp:101` allocates `mem_size * (1 + n_rs_seq)`. On top of the pool's 354
  MiB/card, that is most of the free-VRAM collapse seen before the hang, with `common_fit_params`
  refusing to back anything off under tensor split.
- Where the remaining MTP tax probably lives (H18): 43.4 ms/token at 0.26 acceptance over 6 drafts is
  ~2.5 tokens per step, so ~111 ms/step, of which the verify pass is ~28.7 ms - leaving ~13.7 ms per
  draft replay for a 1-layer draft. That is fixed cost, not compute, and it smells like the same
  host-side/dispatch tax E039 measured, once per draft token.

## Next

1. pp regression: pool=0 on `36ec37826`, then a `-ub 512/2048` sweep. If it is meta-side, the fix is
   in the subgraph/re-init path, not in H9.
2. Memory: the two knobs that matter at 245760 are f16 pool storage (halves 354 MiB/card, moves
   the golden) and `n_rs_seq`, which costs 648 MiB more than anyone priced. If the server stays
   where it is, sizing the pool from the fit result beats either.
3. P2 (host scan + `blk_cells`/`bias` uploads) still open, priority depends on what the traces say
   about the remaining non-kernel time.
4. The dev-box gates keep using `-sm none`, so any future pool change must also be A/B'd under
   `-sm tensor` - this hole was invisible on 1 device only because the *invalidation* path, not the
   data path, was wrong.
5. H18 (the MTP tax) plus two test gaps this addendum exposed: teach the rollback harness to drive a
   `seq_rm(p_keep, -1)` truncation (needs `cparams.n_rs_seq > 0`, which is how the real path arrives)
   and a `PARTIAL_ONLY` restore, so both doors are covered rather than just the whole-state one.