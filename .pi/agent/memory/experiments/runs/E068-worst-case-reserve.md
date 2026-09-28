# E068 - worst-case reserve: does one big reservation remove the ratchet?

- date: 2026-09-28
- machine: dev-rx9070-16g (1x gfx1201), ROCm 10.0.0
- tier: T1
- status: running
- parent: E067 (the slack route faults the GPU), E066 (trips are size-only, one per 256-token padding), E065
  (what the trips cost), H19
- raw: [results/E068-worst-case-reserve/](../results/E068-worst-case-reserve/)

## question and what decides it

`ggml_gallocr_needs_realloc` trips when a tensor's allocation size exceeds the reserved size, and E066 pinned
the rate to exactly the 256-token padding step. The gallocr accepts any graph *smaller* than the reservation
(`ggml-alloc.c:1021` is `size_max >= node_size`), so one reservation covering the worst case should end the
ratchet for the whole session, decode half included. E067 tried to fake that by padding `size_max` and faulted
the GPU; this builds the reserve graph with the *real* worst-case shape instead.

Mechanism: `graph_reserve` builds a synthetic ubatch whose `n_tokens` drives both the batch dimension and,
through the empty memory state, `n_kv`. Shifting its positions to the end of the context (`pos_max = n_ctx-1`)
gives `n_kv = n_ctx` at the real batch width - the true worst case `[n_ctx, n_ubatch]` for the mask and the
n_kv-proportional block keys, rather than the quadratic `[n_ctx, n_ctx]` that passing `n_ctx` as `n_tokens`
would reserve.

Deciding metric: `sched:realloc_size` calls and the prefill wall from the `[prof]` table, plus the same in a
pure decode arm, peak VRAM, and the golden PPL.

Predictions, written first: reallocs fall from 34 to <= 2 in the prefill arm and to 0 in a decode-only arm;
the prefill wall drops by the overlap the syncs forbid (~0.4 s); peak VRAM rises by the mask + block-key
delta only (~64 MB at 32k, not ~1 GB); **the golden PPL stays bit-identical**, because the reserve graph is
built and sized but never executed.

Falsified by: PPL moving; reallocs not falling; VRAM rising far beyond the `[n_ctx, ub]` prediction (which
would mean the quadratic shape is being reserved after all); or an abort in the memory path from the shifted
positions.

## arms

Gate: `LLAMA_RESERVE_WORST_CASE` (default off, so the control is the current behaviour).

| arm | env | reps |
| --- | --- | --- |
| control | - | 3 |
| wc-pp | `LLAMA_RESERVE_WORST_CASE=1`, sparse corpus, `-n 1` | 3 |
| wc-tg | `LLAMA_RESERVE_WORST_CASE=1`, sparse corpus, `-n 256` | 2 |
| control-tg | -, same | 2 |

Plus one golden `llama-perplexity` run per arm, and peak VRAM sampled from sysfs during every run.

## commands

[`results/E068-worst-case-reserve/commands.sh`](../results/E068-worst-case-reserve/commands.sh)

## results

Iteration 1 (shift the reserve ubatch's positions to the end of the context) changed nothing at all.
Iteration 2 (also bump `n_tokens` to `n_batch`, because the reserves that fire here are called with
`n_tokens = 1`) moved VRAM and nothing else:

| iteration | arm | reallocs | `realloc_size` | prefill wall | peak VRAM | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 1, shift only | control | 34 | 1681-1685 ms | 5494-5786 ms | 7838 MiB | 263100.7437 |
| 1, shift only | worst-case | 34 | 1692-1697 ms | 5512-5547 ms | 7838 MiB | 263100.7437 |
| 2, shift + width | control | 34 | 1695.31 ms | 5736.67 ms | 7837 MiB | 263100.7437 |
| 2, shift + width | worst-case | 34 | 1701.35 ms | 5488.45 ms | **8367 MiB** | 263100.7437 |

So the reservation really did grow (+530 MiB) and the reserve graph is provably never executed (PPL
bit-identical), but the trips are untouched: **what grows past the reservation is the mask's first dimension,
and that is not sized by the reserve ubatch at all.**

Why, from the code: the mask's `n_kv` comes from `llama_kv_cache::get_n_kv` (`llama-kv-cache.cpp:1260`), which
reads the KV cache's own cells (`cells.size()`, `cells.used_max_p1()`) - not the ubatch's positions. The reserve
builds its graph without applying its synthetic ubatch to those cells, so the reservation is sized for the
*real* n_kv at reserve time and is outgrown by the next padding step, which is exactly the ratchet E066
measured. The four `mctx->get_n_kv()` sites are `llama-graph.cpp:36,55,861,881`.

Iteration 2's code is not in the tree (the width bump alone removes no reallocs and costs 530 MiB); it is in
`stash@{0}`, and its logs are `it2-*.log` here.

## reads against

- **Established:** the gallocr accepts smaller graphs, so a reservation that is genuinely large enough will end
  the ratchet; the reserve path honours a wider `n_tokens` (the +530 MiB proves the plumbing reaches the mask's
  second dimension); and none of this touches numerics.
- **Not established:** the `n_kv` half. It needs an explicit floor for the reserve path - a reserve flag in
  `llm_graph_params` plus the four `get_n_kv()` sites - not an allocator change.
- The width bump alone is not worth keeping, so the gate stays default-off and inert.
- Consequence for H19: "keep the worst-case budget" is the right lever but it has *two* dimensions, and the one
  that matters cannot be reached from the reserve ubatch today. Budget accordingly: this is a graph-params
  plumbing job, not a padding tweak.