# E023 - one pass, no divisions: cumulative tg +9.8% @40k, +26.0% @164k

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- code: `src/llama-memory-hybrid-idx.cpp`, continuation of the E022 fast path
- expected more; got ~1% per step. **Where the remaining cost is, and why to stop here, is the
  useful part of this record.**

## what changed

1. **Fused verify + write into one pass, and used the `used` set for the run bounds.**
   `llama_kv_cells` keeps `std::set<uint32_t> used`, so `used_min()`, `used_max_p1()` and
   `get_used()` answer "is the used region one contiguous run" in O(1) (`j1 - j0 == nu`) instead
   of by scanning for holes. It also turned out that `is_empty(i)` *is* `pos[i] == -1`, so the
   E022 loop was reading every cell's position array entry twice.
2. **Removed the integer divisions.** The loop had `p/r` and `p%r` per cell with a runtime `r`;
   positions step by one, so a bucket/slot counter pair advances instead.

## measurement, one binary per row, `-r 4` / `-r 2`

| state | tg @ d40960 | tg @ d163840 | pp @ d40960 |
|---|---|---|---|
| before E022 (general scan)      | 181.80 +/- 0.78 | 99.34 +/- 0.01  | 6054 +/- 95 |
| E022 (two-pass fast path)       | 196.82 +/- 0.52 | 121.19 +/- 1.27 | 6043 +/- 92 |
| + fused single pass             | 199.33 +/- 1.45 | 123.70 +/- 0.32 | 6064 +/- 109 |
| + divisions removed (**final**) | **199.57 +/- 0.80** | **125.19 +/- 1.22** | 6089 +/- 99 |

Step-time view at 164 k: 10.07 ms -> 8.25 -> 8.08 -> 7.99 ms. The original scan was measured at
about 3.8 ms there (E021), so ~2.1 ms is gone and **~1.7 ms of per-step host work remains inside
this function**.

Both refinements are worth only 1-2%, not the ~2x I predicted. Correctness was re-checked after
each: golden, sparse and dense `-c 8192` arms bit-identical throughout (262938.7619 / 267157.2589 /
267157.4202), `np 2` inside its own ~1e-7 spread, and the sparse arm composes (tg 199.57 dense vs
199.4 sparse at 40k).

## why micro-optimizing this loop is finished

What the fast path must now write per step is roughly `n_kv` ints into `cell_blk`, `n_kv` ints into
`blk_cells`, `4*n_blocks` ints into `blk_pos` plus a fill - at 164 k that is about 3 MB of scattered
scalar stores, and ~1.7 ms is ~3 ns per store, i.e. the writes are leaving L1/L2 and that is the
floor for *any* implementation that rebuilds the mapping every step. Loop shape, divisions and pass
count are no longer the constraint.

Consequence for the memoization idea (E021's option "cache the mapping"): it does not help by
making the loop cheaper, it helps only if the previously written arrays are **reused instead of
rewritten** - keep them in the memory object, patch the O(r) cells that a new token touches, and let
the async host->device upload carry the unchanged bytes. That is the remaining ~1.7 ms at 164 k
(about 20% of the step) and it needs a sound invalidation signal for evictions, defrag and context
shift, which is exactly the part that must be looked at carefully rather than guessed.

The other ~1/3 of the original depth slope is GPU-side O(n_kv) work (mask build, `set_rows`,
`get_rows` expansion, `top_k`) and is untouched by any of this.
