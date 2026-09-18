# E024 - what the per-QSA-layer GPU work actually is (analysis, not measurement)

- date: 2026-09-17 | machine: dev-rx9070-16g | tier: T3 analysis | status: open
- source: `src/models/qwen4exp.cpp:596-700` (`build_qsa_top_k`), `:700-790` (`build_attn_qsa`)
- shapes are read off the graph code; **no byte figures below are measured**. The point is to say
  where to look before spending a build on it.

## the per-layer, per-step work at n_kv = 40960, r = 4, idx_dim = 128, f16 cache

| step | nodes | bytes | note |
|---|---|---|---|
| `index_k_proj` mm + `cpy_k` | 2 | ~1 MB | writes 128 raw values for the new token |
| `get_rows(k_all, blk_cells)` | 1 | **~21 MB** | reads *and* writes the whole indexer cache: 40960 x 128 x 2 B x 2 |
| `r` x `cont(view)` + `add` pooling | ~8 | **~29 MB** | 4 passes over 10240 x 128 plus the sums |
| `scale`, `rms_norm`, `rope_multi` | 3 | ~15 MB | all over the pooled [n_kv/r, 128] tensor |
| `index_q_proj` mm, norm, rope | 3 | ~3 MB | weights dominate |
| `mul_mat(pooled, q)` | 1 | ~3 MB | reads pooled, writes 10240 x 4 |
| `relu`, 4-head `cont`+`add`, bias | ~9 | ~1 MB | |
| block scores -> cells: `permute`/`cont`, `get_rows(cell_blk)`, `cont`/`permute` | 3 | ~1 MB | writes n_kv values |
| mask `cast` + `add` | 2 | ~0.3 MB | f16 -> f32 over n_kv |
| `top_k(expanded, 2051)` | 1 | ~0.2 MB read | runs on GPU (`ggml-cuda.cu:5456`), not host |
| **indexer total** | **~33** | **~76 MB** | |
| attention, dense | 1 | 42 MB | 40960 x 2 heads x 256 x 2 x 2 B |
| attention, sparse (E020) | 1 | ~4 MB | 2051 rows instead of 40960 |

Two things fall out of this. **The indexer path is about twice the bytes of dense attention and
roughly nineteen times the bytes of sparse attention**, so after E020 the per-layer QSA cost is
dominated by work that has nothing to do with attention. And it is **~33 graph nodes per QSA layer
per step**, which on the real model is 12 x 33 = ~400 nodes per step for the indexer alone - the
per-step launch and graph-walk cost E011 measured is the same class of thing, and this contributes
to it 12 times over where the dummy contributes once.

## the obvious target

The gather plus the four pooling passes (~50 of the ~76 MB, ~9 of ~33 nodes) exist because the
indexer caches **raw per-token keys** and re-derives pooled block keys every step, and it must do
that as written because pooling precedes norm and rope (`qwen4exp.cpp:599`).

But blocks are aligned runs of `r` positions, so a block's pooled, normed and roped key can never
change once its `r` tokens are written. Caching block keys in the indexer cache - a second, r:1
coarser cache, appended to as blocks complete, with only the trailing partial block recomputed -
would cut the per-step indexer read from `n_kv x 128` to `n_kv/r x 128` and delete the pooling
passes outright. That is the GPU-side counterpart of what E022/E023 did on the host, and unlike
that change it **does** scale with layer count: 12x on the real model.

Costs to weigh before attempting it: a second cache means a second set of append / rollback /
context-shift paths in the memory layer, and the block cache is only valid if block boundaries are
stable under eviction and defrag - the same questions E023 left open for host-side memoization, but
now with device memory and a `cp_k`-like copy per block completion.

## what would confirm the picture (not done)

- **2-QSA-layer dummy**: build with `--layers 8` (2 full-attn layers) or 4 layers with
  `full_attention_interval=2`, and compare the residual tg slope per QSA layer against the 1-layer
  case. This also settles the per-step vs per-layer question E023 could not, for the host fill.
- **Node count**: the graph op list per decode step is dumpable; ~33 nodes/QSA-layer is a reading
  of the builder, not a count of a real graph.
- Nothing here has been profiled on the GPU; `test-export-graph-ops` exists (`tests/`) and exports
  shapes for a full ubatch (prefill), which is not the decode case we care about.
