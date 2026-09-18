# E036 - the QSA chain costs 3.3x what sparse decode saves, and it selects in the wrong order

Prompted by the E035 addendum: sparse decode engaged on the bench and moved `tg` by nothing. To find
out what decode's context-proportional cost actually is, `tools/qsa-no-indexer.patch` adds
`Q4EXP_NO_INDEXER=1`, which drops `build_qsa_top_k` from the graph so attention runs dense-causal. The
output is wrong on purpose - on the dummy there is no quality signal at all, so this is a cost probe.

Dev box, dummy (one QSA layer of four), single card, tg128, r=3:

| depth | indexer on, dense decode | indexer on, sparse decode | indexer off, dense attention |
|---|---|---|---|
| 4096   | 239.00 | 237.86 | 256.77 |
| 40960  | 199.86 | 207.54 (+3.8%) | 239.09 (+19.6%) |
| 163840 | 125.80 | 140.44 (+11.6%) | 193.36 (+53.7%) |

Same numbers as ms/token, and the two differences that matter:

| depth | step, indexer on | sparse decode saves | indexer chain costs |
|---|---|---|---|
| 4096   | 4.18 ms | 0.02 ms | 0.29 ms |
| 40960  | 5.00 ms | 0.19 ms | 0.82 ms |
| 163840 | 7.95 ms | 0.83 ms | **2.78 ms** |

Per QSA layer per decode step at 164k, the chain that *decides* which KV to read costs 2.78 ms, and the
saving from reading only 2051 of 163840 rows is 0.83 ms. That ratio - about 3.3 to 1, growing with
depth - is the whole explanation for the bench null: 12 layers of chain on a 45 ms step swamps a few ms
of avoided KV read, whichever kernel does the avoiding. It also retires my +20-30% prediction for
sparse decode on that box; the arithmetic was right about the bytes and wrong about which bytes were
on the critical path.

The chain's cost does not look bandwidth-bound: 84 MB of gathered keys in 2.78 ms is 30 GB/s, and there
are ~33 nodes per layer per step, so ~85 us each. Small dependent kernels, not a big read.

## Why the chain is bigger than it needs to be

`src/models/qwen4exp.cpp:655-686` does the selection in the opposite order to the architecture:

```
 score (blocks, n_kv/4)  ->  expand to cells (n_kv, f32)  ->  + mask  ->  ggml_top_k over n_kv cells
```

Report Eq. 16 and 19 select `K_B = ceil(K/r) = 512` **blocks**, then expand. Because every cell of a
block carries the block's score, top-k over the expanded cells returns the same set, so the current
order is not wrong - it is just 4x the data through the largest terms in the chain: two
`cont(permute(...))` copies of an `n_kv x n_tps` f32 array, an add over it, and a top-k over `n_kv`
inputs where `n_kv/4` would do. Selecting at block level and expanding only the 512 winners deletes
both full-size copies and shrinks the top-k input 4x, and it matches the paper's formulation instead
of diverging from it. That is maybe 15 lines inside one function, and unlike H9 it needs no new cache,
no position invalidation, and no touch to `llama_memory_hybrid_idx`.

So the order of work is: block-level selection first (cheap, contained), then decide whether the
remaining per-layer gather and pooling still justify H9's cache. Sparse decode stays as it is - it
pays on a single card and is invisible behind the chain on four - and the vec-kernel port is not worth
~150 lines plus a doubled instantiation set while the chain is the largest term.

## Open

- whether the 2.78 ms scales linearly with layers on the real model, and how tensor split places the
  chain: the E035 gate line shows 24 Q heads and 2 KV heads *per device*, which would mean attention
  is replicated rather than split there, and that needs checking in the fork;
- what fraction of 2.78 ms is `ggml_top_k` alone - the block-level change answers it implicitly;
- whether `no_indexer` on a real checkpoint degrades quality (expected: yes, badly - it is the design,
  and the dummy cannot say).
