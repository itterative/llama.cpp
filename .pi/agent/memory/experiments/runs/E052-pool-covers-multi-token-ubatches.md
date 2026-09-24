# E052 - the pool now covers multi-token ubatches, with prefill behind a flag

Commits `31f19dc94` (reuse predicate fix) and `bd0b294a8` (width policy). Raw:
[results/E052-pool-widths/](../results/E052-pool-widths/) (`F`..`I`, dev box, same session,
`GGML_PROF_REGIONS=1`).

## Why the width changed

E050/E051 left the pool restricted to `n_tokens == 1` to kill the prefill regression. That was too
broad: under `draft-mtp` the target advances in `n_draft + 1`-token jumps, so a verification pass -
the widest single consumer of block-key derivation there is - got no pool at all, and since no step
pools, the pool is never even populated. Every +30% number quoted so far came from `llama-bench tg`,
which is single-token decode, i.e. not the deployment being tuned.

The condition is now `ubatch.n_tokens >= 16 && Q4EXP_POOLED_NO_PREFILL`, so the default pools at every
width and one env var keeps it out of chunked prefill. 16 sits above any realistic verification width
(it matches the rtile `n_q` cap from E047 for a different reason, but the coincidence is handy) and
below prefill ubatches, which are `n_ubatch` (512-2048 here).

Naming note: I introduced `Q4EXP_POOLED_DECODE_ONLY` with a `!= 1` test; the user changed the test to
`>= 16`, which made that name wrong (verify widths are now included), so the env became
`Q4EXP_POOLED_NO_PREFILL` before committing.

## Prices, dev box, q4exp-48l-12qsa, -sm tensor -d 16384 -p 8192 -n 128 -b 2048 -ub 1024 -r 2

| arm | config | pp8192 | tg128 | re-reserves |
| --- | --- | --- | --- | --- |
| F | `Q4EXP_POOLED=0` | 704.57 | 32.48 | none |
| G | pool, all widths (**new default**) | 683.92 | 33.20 | 24 + 7 size, 18.2 s |
| H | pool, `n_tokens != 1` gate (old) | 702.71 | 33.05 | 1 @ 24 ms |
| I | pool + `Q4EXP_POOLED_NO_PREFILL` | **705.67** | **33.51** | 1 @ 24 ms |

So arm I keeps prefill whole (matches F within noise) and keeps the decode win, at the cost of one
24 ms re-reserve per pass - the cold start, since prefill leaves the pool empty. G is what "pool
everywhere" costs: -2.9% pp and 31 re-reservations, because at prefill widths the saved derivation is
small next to expert traffic while the extra tensors keep making `ggml_gallocr` re-reserve.

The tg column must be read loosely: H and I differ by 1.4% while being identical workloads for this
benchmark (llama-bench has no spec decode, so no ubatch ever lands in the 2..15 band), so any decode
delta under ~2% here is run variance, not effect.

## Correctness

Identical everywhere: golden `-sm none` **263113.6984** (pool off, default on, and with the flag);
deep sparse `-c 8192 -sm tensor` **267035.1320** in all three - which is the check that matters for
this change, since it exercises pooled multi-token prefill and says it matches the historic path
exactly; rollback harness `cksum=4903597782947487587`, `rollback_replay_mismatch=0`;
`test-save-load-state` 7 PASS.

The reuse-predicate bug (`new_cells->ne[0]` checked against `ratio` instead of `ratio*n_new`, `ne[1]`
against `n_new` instead of `n_stream`) was latent only because the old gate pinned `n_new` to 1; with
wider pooling it would have blocked graph reuse on every step.

## Still unmeasured

Nothing here prices the case the change exists for. `llama-bench` cannot drive `draft-mtp`, so the
verification pass has never been timed pooled or not. On the bench box, with the server or
`--spec-type draft-mtp`: `tg` at 131k with `Q4EXP_POOLED=0/1` and `-v` to count `qsa pool: mode` lines -
if the mode is stuck at the cold-start rebuild rather than reaching cached steps, the run is paying the
derive-all on every single step and the default is wrong for MTP, not just for pp.