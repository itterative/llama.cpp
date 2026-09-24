# E049 - the pp regression is ggml-alloc re-reservations, and my cost model for it was wrong

> Full independent audit: [reviews/E049-review-of-h9-prefill-churn.md](../reviews/E049-review-of-h9-prefill-churn.md)
> (`reviewer-6`, asked to falsify my claims). Read that file before re-deriving any of this - it names
> the code lines that killed two of my claims. This record keeps only what survived.

E045 left `pp8192` at -16..-22% with `Q4EXP_POOLED=1`, unexplained, with the note "needs >1 device".
It reproduces on one card: `q4exp-48l-12qsa`, `-sm tensor -fa 1 -lzm on-direct -d 16384 -p 8192 -n 0
-b 2048 -ub 1024`, pool off 706 vs pool on 684 (**-3.2%**). Smaller magnitude than the 4-card box, same
direction, so the mechanism was local and measurable.

## What is actually happening

With the region profiler (E047's tooling, `02d963eb0`), the entire pp delta sits in `graph:alloc`:
22 ms/ubatch without the pool, 512 ms with it, and `sched:realloc_*` accounts for 510 of those 512.

- The graph that `sched_reserve()` measures is **not** the graph the pooled build emits. The measure
  pass runs with an empty cache, so `qsa_pool_get` bails on `nu == 0` and returns mode NONE - in both
  arms, which is why the measure print is identical for pool on and off (`nodes = 6775, leafs = 1358,
  need = 1134.82 MiB`). At runtime the pooled graph has 6813-6825 nodes.
- `ggml_gallocr_needs_realloc` compares `n_nodes` first, and with multiple buffers `alloc_graph`
  returns false instead of growing in place. So the first pooled ubatch re-reserves, and the new
  reservation is sized to *that* ubatch, discarding the worst-case headroom. After that, every
  depth-proportional tensor growth trips it again: observed need ratcheting 377 -> 652 MiB, about
  6.3 MiB per 1024-token ubatch, with `nodes` constant at 6813.
- The mask is **not** the main grower (I claimed it was). Biggest is the pre-sum `score` tensor
  `[n_blocks, 4, 1024, 1]` f32 at +4 MiB/ubatch; the `Meta(ROCm0)#attn_inp_kq_mask#0` copy is second at
  +2 MiB. So a fixed-size mask would not have fixed this.
- `meta:subgraph` count is identical between arms (3088), so this is not about subgraph count at all.

## What I got wrong, and how it was caught

Two claims died, both in the "cost" half of the story:

1. **"A re-reserve costs ~575 ms"** - wrong. It is a full device `synchronize` of every backend plus
   ~16-24 ms of real host work; the 575 ms was *the previous ubatch's device time arriving under a
   different label*, because the pool moves the mandatory drain from inside `graph:compute` to the top
   of `graph:alloc`. Nothing is freed or reallocated at all: the physical buffer is 1134 MiB and the
   runtime need never exceeded 652 MiB, so the realloc branch in `ggml-alloc.c` cannot fire.
   Confirmed directly after the fix: `sched:realloc_buft 4 calls 23.4 ms/call` with min..max
   22.77..24.24 - a re-reserve is ~23 ms.
2. **"The pool saves 29% of prefill device work"** (`graph:compute` 1209 -> 860 ms/ubatch) - not
   supportable. Those regions tile wall time, so only the *sum* is comparable across arms: 1245 vs
   1386 ms/ubatch, i.e. the pooled arm was 11% slower in that sample. Worse, I had compared the
   `[prof] warmup` table against a `pp8192` t/s measured in the *timed* reps at a completely different
   depth range - different populations. So whether pooling can win in prefill is **still open**, not
   "real but eaten by churn" as I reported.

Also corrected: `-sm none` and `-sm tensor` both have `n_buffers == 2` (the CPU backend's buft is
always present), so the single-buffer/multi-buffer split I used to explain why `-sm none` churns
harmlessly is wrong. Why `-sm none` + pool-off triggers 24 re-reserves at all is still unexplained.

Method lesson: my own probe had a live hazard - calling `ggml_gallocr_reserve_n_size` before the sync
loop can free a device buffer with work in flight. Removed. And `graph:alloc`'s new **min** column is
what exposed the relabeling: min 38.5 ms against a mean of 512 ms cannot be a constant allocation cost.

## The change (`qsa_pool_get`, uncommitted at time of writing)

Restrict the pool to single-token ubatches, i.e. decode. Placed in `qsa_pool_get` next to the existing
`n_stream != 1` bail-out rather than in the model builder, because the predicate is consulted from both
`can_reuse` and the graph build, and gating only one would desynchronise the reuse check from the graph
that gets built.

| dev box, d16384 | before gate | after gate |
| --- | --- | --- |
| pp8192 | 705.54 -> 704.47 (-0.15%) | was -3.2% |
| tg128 | 32.17 -> 33.10 (**+2.9%**) | win kept |
| re-reserves | 16/pass @ ~25 ms avg | 4 total @ 23.4 ms |

Correctness after the gate, all identical to the pre-gate values: `rbtest` rollback harness
`cksum=4903597782947487587` and `rollback_replay_mismatch=0` in both arms; `test-save-load-state` 7
PASS with the pool on; golden `-sm none` **263113.6984** both arms. Note for future comparisons: the
golden with `-sm tensor` is **263113.7846** in both arms - the 0.086 shift is split-mode reduction
order, not a regression.

## Open

1. Whether pooling prefill can win at all, measured without the reservation confound. The reviewer's
   route (its d1 + d2, ~40 lines, no ggml change) is: collapse REBUILD into "CACHED with `wm = 0`" by
   feeding the block scores from the `set_rows` result instead of the freshly derived tensor - which
   also removes an ordering-only dependency that today relies on `build_forward_expand` happening to
   precede the `mul_mat` - and then make the measure pass build that same shape, exactly as
   `llama-memory-hybrid-idx.cpp:1015` already does for the sparse-mask case. Zero re-reserves, both
   wins. Untried.
2. The general trap is worth reporting upstream: any graph whose node count differs from the measured
   worst case silently discards its headroom under multi-buffer split and then re-reserves per ubatch
   forever. Cheap to detect with `GGML_PROF_REGIONS=1` once the `sched:realloc_*` counters are in.
3. The 4-card box showed -16..-22% where one card shows -3.2%; the direction matches, the magnitude
   does not. Still needs a `Q4EXP_POOLED=0/1` pp pair on the bench box with the gate in place.