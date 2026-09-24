# E051 - collapsing REBUILD into CACHED(wm=0) is a win; reserving the pooled shape is not

Follows E049 (mechanism) and E050 (4-card confirmation of the decode-only gate). d1 is `8206d79d8`
(`llama : one qsa pool graph instead of two`); d2 was tried and reverted within this record's session.
Dev box,
`q4exp-48l-12qsa` + `q4exp-4l`, `-sm tensor -fa 1 -lzm on-direct -b 2048 -ub 1024 -r 2`,
`GGML_PROF_REGIONS=1`, same session for all arms. Raw:
[results/E051-pooled-prefill-a-b-c/](../results/E051-pooled-prefill-a-b-c/) (A/B/C, the d1+d2 attempt) and
`[results/E051-d1-final/](../results/E051-d1-final/) (D/E, d1 alone).

## The two changes tried

**d1 - one pooled graph shape.** REBUILD existed only to say "the pool has nothing usable yet", and it
built a different node set than CACHED to do it (derive-all-and-write vs read-and-derive-new). With
`wm = 0`, CACHED already *is* that: it derives and writes `[0, n_bid)`. So REBUILD is deleted, and the
scores now read the result of `ggml_set_rows` instead of the pool view or the freshly derived tensor -
`set_rows` returns a view of its destination, so that gives a real graph edge where the previous code
relied on `build_forward_expand` happening to be emitted before the `mul_mat`. Net **-8 lines**, one
mode instead of two, and the cold-start `n_bid == 0` case returns NONE explicitly (the historic chain
must supply every column when no block is complete).

**d2 - make `sched_reserve` measure the pooled shape.** Reverted; see below.

## Result of d1 alone (final state)

| d16384, same session | pool off | pool on, gated |
| --- | --- | --- |
| pp8192 | 704.52 ± 1.57 | 703.78 ± 0.43 (**-0.1%**) |
| tg128 | 32.63 ± 0.30 | 33.07 ± 0.01 (**+1.4%**) |
| `sched:realloc_buft` | never | **1 call @ 24.33 ms** |
| `sched:realloc_size` | never | 0 |

Re-reservations went 6 -> 1 against the pre-d1 build, and the single remaining one is the honest
minimum: the reserve pass runs on an empty cache, so `qsa_pool_get` cannot report a pooled mode and the
tg measure graph keeps the historic shape; the first real decode step differs and pays 24 ms once.

Correctness, all unchanged from before the refactor: golden `-sm none` **263113.6984** both arms;
rollback harness `cksum=4903597782947487587`, `rollback_replay_mismatch=0` both arms;
`test-save-load-state` 7 PASS with the pool on.

## Result of d1+d2 (why d2 is not in)

d2 answered `qsa_pool_get` from an empty cache with the steady-state pooled shape, so the measure graph
became 6825 nodes instead of 6775. Priced against the same two arms:

| | pool off | gated (d1+d2) | pooled prefill (d1+d2) |
| --- | --- | --- | --- |
| pp8192 | 705.34 ± 0.39 | 694.37 ± 15.46 | 692.11 ± 15.47 |
| tg128 | 31.82 ± 0.10 | 33.08 ± 0.01 | 32.92 ± 0.17 |

Both pooled arms show the same 8 `sched:realloc_size` calls at **535 ms each** in their pp tables (4.28 s
of a ~11 s pass), which the pre-d2 gated build did not have. Cause: d2 changed what the reservation is
built for, so the *gated* runtime prefill graph (historic, 6775) no longer matched it - the mismatch
moved rather than went away, and the tight re-reserve then ratchets with depth as in E049.

So d2 is only coherent together with pooled prefill, and pooled prefill is not worth it: C vs B is
692.11 vs 694.37, inside the ±15 spread - **no gain**. Which closes E049's open question in the
negative: with the topology problem fixed, pooling prefill still buys nothing measurable here. The
prefill pass is dominated by MoE expert traffic; the indexer chain it saves is small next to it, while
the depth-proportional input tensors (`bias`, `blk_cells`, the `kq_mask` copy) keep the reservation
moving. The decode-only gate stays, and the pool ships as a long-context decode feature.

## Caught by the assert build

Reverting d2 initially left the `nu == 0` test deleted along with the branch that used it, so the
reserve pass fell through to `cells.pos_get()` on an empty cache and died on
`llama_kv_cells.h:392: Assertion 'pos[i] != -1'`. A release build would have read past the end of the
cell array instead of stopping. Keep gates running against `-O2 -g3` without `-DNDEBUG`.

## Still open

- 24 ms once per pass for the decode reservation mismatch. Fixing it means teaching `sched_reserve` to
  measure a shape chosen from live state it does not have; a cheaper alternative is to reserve the tg
  graph with the pool *forced* on, accepting a `wm = 0` cold start - untried, and 24 ms is small.
- E050's missing d131072 pool-off pair on the 4-card box.
- `graph:set_inputs` ~100 ms per 1024-token prefill ubatch, identical in both arms, unexamined.