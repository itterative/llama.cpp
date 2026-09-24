## Bottom line

Three of your seven claims survive. **C2 is right** (and I can give you the exact line that makes it true). **C3 and C4 are wrong in the specific mechanism** - the none/tensor asymmetry is not single-vs-multi-buffer, and the re-reserve path **cannot be freeing/reallocating your compute buffers at all**, which removes the load-bearing wall of your story. **C6 is right to be worried**: the -29% is not a measurement you can bank, and the reason is worse than "async entry point" - you quoted the *warmup* table, which is a different population from the `pp8192` number it is being compared against. And your own data contains a number (`graph:alloc` min = 38.5 ms) that bounds the re-reserve's real host cost at ~16 ms.

Most important: **`llama-bench -v` on your current binary already prints the answer to C2/C3/C5/Q2/Q4**, because your build has `NDEBUG` off and llama-bench's `-v` skips the null log callback. Details in "Minimum decisive experiment".

---

## C1 - CONFIRMED, with a caveat that matters

`src/llama-context.cpp:614` (`mctx = memory->init_full()`), `:596-597` (`n_tokens = min(n_ctx, n_ubatch)` = 1024, `n_seqs = n_seq_max` = 1). The full-cache ctor sets `n_kv = kv->get_size()` (`src/llama-kv-cache.cpp:2698-2700`), and the indexer cache is built with the same `kv_size` as attention (`src/llama-memory-hybrid-idx.cpp:76-79`), so `build_qsa_top_k`'s `n_kv`/`n_blocks` are also worst-case. Two precisions:

- Note `n_ctx` here is **24576**, not 16384: `tools/llama-bench/llama-bench.cpp:1302` sets `cparams.n_ctx = n_prompt + n_gen + n_depth`. Your deepest runtime ubatch also reaches 24576, so the reserve is exactly tight, not generous.
- The thing being sized is not the mask *leaf* (it is host-resident with `data != NULL`, so galloc ignores it - `src/llama-graph.cpp:488`) but the **cross-backend copy node** `Meta(ROCm0)#attn_inp_kq_mask#0` created at `ggml/src/ggml-backend.cpp:1406` and inserted as a *node* of `sched->graph` at `:1510-1516`. f16 `[n_kv,1024,1,1]` = 2 MiB per 1024 tokens - that reproduces your 2.097 -> 4.194 -> 6.291 MiB steps exactly (`src/llama-graph.cpp:30-45`).

**Caveat that defeats the practical value of C1:** the worst-case headroom only exists until the first re-reserve. `ggml_gallocr_reserve_n_impl` overwrites `node_allocs[i].dst.size_max` from the *current* graph every time (`ggml/src/ggml-alloc.c:856-911`). So one topology-triggered re-reserve at n_kv=1024 replaces the 48 MiB budget with a 2 MiB budget, and every subsequent ubatch trips `size_max >= node_size` (`:1006`). That is your ratchet, and it is real.

## C2 - CONFIRMED, and here is the exact reason

`ggml_gallocr_needs_realloc` checks `galloc->n_nodes != graph->n_nodes` first (`ggml/src/ggml-alloc.c:1010`), then leafs (`:1017`), then per-tensor sizes. `ggml_gallocr_alloc_graph` returns **false** for `n_buffers > 1` (`:1052-1066`, and the header comment at `ggml/include/ggml-alloc.h:70-72`). 6813 != 6775 -> false -> the re-reserve branch at `ggml/src/ggml-backend.cpp:1612`.

The corollary is confirmed with a precise mechanism: `qsa_pool_get` early-returns a default `llama_qsa_pool` whose `mode = NONE` (`src/llama-memory-hybrid-idx.h:26`) because at reserve time the live indexer cells are empty - `if (nu == 0 || j1 - j0 != nu) return res;` (`src/llama-memory-hybrid-idx.cpp:356-362`). It reads `mem_idx->get_cells(...)` - the **live cache**, not the context snapshot - so `init_full()` cannot help it. Both arms therefore measure the identical NONE graph. That is why the prints match.

**But your instrumentation's label is lying to you.** `sched:realloc_ids` vs `sched:realloc_grow` is chosen by `backend_ids_changed` (`ggml/src/ggml-backend.cpp:1615`), which compares *buffer-type assignments* per node index (`:1593-1609`), **not** topology. Your code comment says "ids changed = the graph topology differs from the previous ubatch" - that is not what the variable means. The 6-vs-2 split does not decompose into "grow vs topology"; both categories are topology+size events, and a node-count change can *itself* flip `backend_ids_changed` because the arrays are compared index-wise over the new count.

## C3 - WRONG in the part that carries your mechanism

The ggml-alloc half is quoted correctly but incomplete: `needs_realloc` also fires on **any per-tensor size growth** (`ggml-alloc.c:1024-1046`), not just topology.

The load-bearing half is false: **`n_buffers` is 2 in both split modes.** `src/llama-context.cpp:406-422` pushes one entry per backend, and `backends` is always `[device(s)..., CPU]` (`:330-357`), with the CPU slot's buft replaced by the first device's *host* buft (`:412-418`). So `-sm none` = `[ROCm0 dev buft, ROCm0 host buft]`, `-sm tensor` = `[Meta buft, ROCm0 host buft]`. Neither is single-buffer, so neither ever takes the "grow in place" branch - which your own data already shows: `-sm none` re-reserves 24 times.

The real none/tensor asymmetry is elsewhere, and it is not mirror width:

1. `ggml_backend_meta_device_supports_buft` returns **false for any non-meta buft** (`ggml/src/ggml-backend-meta.cpp:162-166`). So under `-sm tensor` every host input becomes a device copy node living in the compute buffer.
2. But note: `ggml_backend_cuda_device_supports_buft` only accepts the HIP *host* buft when the device is `integrated` (`ggml/src/ggml-cuda/ggml-cuda.cu:5593-5597`). Your RX 9070 is not, so `-sm none` copies too. **So copies are not the asymmetry either** - I could not find a code-level reason why `-sm none` pool-off triggers 24 times while `-sm tensor` pool-off triggers 0. That is a genuine gap in your evidence and mine; the `-v` probe below names the tensor and reason directly.
3. `pipeline_parallel` is off in all four arms (`src/llama-context.cpp:429-433` requires `n_devices() > 1` and `SPLIT_MODE_LAYER`), so `n_copies == 1` and the alternating-copy theory is dead. Good - one less variable.

## C4 - WRONG. No buffer is freed or reallocated.

`ggml_gallocr_reserve_n_impl` reallocates a buffer **only when a chunk's new max exceeds its current physical size** (`ggml/src/ggml-alloc.c:913-946`, condition at `:921`, and `ggml_vbuffer_chunk_size` is the physical `ggml_backend_buffer_get_size`, `:412-414`). `max_chunk_size` is `SIZE_MAX` for both the ROCm and Meta bufts (`ggml-cuda.cu:931` -> `NULL` defaults to `SIZE_MAX`; `ggml-backend-meta.cpp:305-310` takes the min over sub-devices), so `n_chunks == 1` and there is exactly one physical buffer per buft.

Your physical buffer is **1134.82 MiB** (the worst case). Your runtime need is **377-652 MiB**. `652 > 1134` is false -> `realloc == false` -> `ggml_vbuffer_free` is never called. **Nothing is freed, nothing is mirrored-width-reallocated, and the 575 ms is not allocation.**

What the 575 ms actually is: `ggml/src/ggml-backend.cpp:1693-1696` -

```c
// synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
for (int i = 0; i < sched->n_backends; i++) {
    ggml_backend_synchronize(sched->backends[i]);
}
```

The Meta device has `event_new = nullptr` (`ggml-backend-meta.cpp:197`), so nothing here is event-granular: it is a full device drain, and it drains **the previous ubatch's in-flight work**. In the pool-off arm that same drain happens inside `graph:compute`, at the split-input sync `ggml/src/ggml-backend.cpp:1739-1746` ("inputs from the user must be copied immediately..." -> `ggml_backend_synchronize(split_backend)` then a blocking `ggml_backend_tensor_copy`). The pool just **moves the wait from `graph:compute` to `graph:alloc`**.

Your own table proves the real host cost is small: `graph:alloc` **min = 38.5 ms** vs pool-off's **mean = 22.25 ms**. The cheapest pooled `graph:alloc` - the first ubatch of the pass, when the queue is empty and the sync returns instantly - costs ~16 ms more than a non-re-reserving `graph:alloc`. That 16 ms is `ggml_gallocr_reserve_n` plus a second `alloc_graph` pass. The other ~490 ms is device wait.

Arithmetic cross-check: 6x575.3 + 2x316.3 = 4.08 s, and 8 x (512.5 - 22.25) = 3.92 s. Within 4%, i.e. **essentially all of the pooled arm's `graph:alloc` is the re-reserve path**, and essentially all of that is the sync.

Consequence: C4's decode asymmetry explanation collapses, though its conclusion survives for a different reason - in decode there is ~5 ms in flight, so the drain is ~5 ms, so 34 re-reserves cost ~nothing and the +30% tg is unaffected.

## C5 - mechanism CONFIRMED, conclusion WRONG

The ratchet is real (see C1 caveat). But "mask first among them" is false, and "a fixed-size mask would flatten the curve" is false. With r=4, n_idx_h=4, idx_dim=128, n_tps=1024 (from `models/q4exp-48l-12qsa.gguf`: `compress_ratios` all 4, `indexer.head_count` 4, `indexer.key_length` 128, `indexer.top_k` 2048):

| tensor | shape | growth per 1024-token ubatch |
|---|---|---|
| `score` (pre-head-sum, `qwen4exp.cpp:830`) | `[n_blocks,4,1024,1]` f32 | **+4 MiB** |
| mask copy | `[n_kv,1024]` f16 | +2 MiB |
| `indexer_score` (post-sum, `:836-846`) | `[n_blocks,1024,1]` f32 | +1 MiB |
| `bias` copy | `[n_blocks,1024,1]` f32 | +1 MiB |
| `members` (NONE/REBUILD only) | `[128,4*n_blocks]` f32 | +0.5 MiB |

The mask is the **second** biggest grower. Fixing its size leaves ~6 MiB/ubatch of growth and the same per-ubatch churn. Your observed 6.34 MiB/ubatch is consistent with this list, not with the mask alone. The only correct part of C5 is "would not stop the first re-reserve" - right, because the first trigger is `n_nodes`.

## C6 - the comparison is INVALID. Do not discard or keep anything on this evidence.

Three independent defects, in order of severity:

**(1) You compared different populations.** `tools/llama-bench/llama-bench.cpp:2416-2419` flushes the counters *after* the warmup and prints them as `[prof] warmup`. So your table covers the warmup pass only: `test_prompt(ctx, 8192)` from a **cleared** cache (`:2388-2400`), i.e. n_kv = 1024...8192. The `pp8192` t/s comes from the *timed* reps, which run at n_kv = 17408...24576 after a 16384-token depth run (`:2420-2470`). The pool's device-side benefit scales with `n_blocks` (n_bid 256...2048 in the warmup vs 4352...6144 timed) while the churn cost does not. You measured the population where the pool looks worst and the churn looks worst, then compared it to a t/s number from the population where both are different.

**(2) The regions tile wall time, so only the SUM is comparable across arms.**

- pool off: 1.27 + 22.25 + 12.34 + 1209.3 = **1245 ms/ubatch**
- pool on: 1.26 + 512.5 + 12.24 + 859.9 = **1386 ms/ubatch**

The pooled arm is **11.3% slower per ubatch** in the warmup. Your `-29%` is a reattribution, not a saving: the drain moved from `compute` (`ggml-backend.cpp:1739-1746`) to `alloc` (`:1693-1696`). This is exactly the failure mode you flagged, and it is worse than you thought because the mechanism is a *full device synchronize*, not just an async boundary.

**(3) The residual is genuinely undetermined by these regions.** With D = device time/ubatch, P = work still pending when `graph:compute` returns, W = real host work in the re-reserve:
`alloc(on) = P + W = 512.5`, `compute(on) = D_on - P = 859.9`, `compute(off) ~= D_off = 1209.3`. Two equations, three unknowns. The `min = 38.5 ms` datum pins `W ~= 16 ms`, which forces `P ~= 490` and **`D_on ~= 1350 ms > D_off`** - i.e. in the warmup the pool is a ~12% device-side *loss*. I can construct no device-side mechanism for that from the code (at n_kv=8192 the write chain derives 256 rows vs the derive chain's 2048), so I do not fully believe solution A; but the data cannot exclude it, and that is the point. **`graph:compute` cannot answer the question you are asking it.**

Also note `graph:set_inputs` is 12.34 vs 12.24 - the pool's host-side watermark/`new_cells`/`new_pos` filling is free. That is a real (small) positive finding you have not claimed.

## C7 - which fix I would defend

Not (a), and not (c) as written.

**(c) as stated is a no-op.** The physical buffer is never shrunk (C4). What shrinks is the per-tensor `size_max` budget. (c) corrected = "keep a high-water mark for `node_allocs[].size_max` across reserves". That would work, but `node_allocs` is indexed by node position and rebuilt wholesale per graph, so a position-keyed high-water mark is only sound when the topology is stable - which is precisely the condition you cannot guarantee. I would not defend this upstream.

**(a) works and is one condition, but put it in the right place:** inside `llama_memory_hybrid_idx::qsa_pool_get` (`src/llama-memory-hybrid-idx.cpp:340`, alongside the existing `n_stream != 1` early-out), not in the model builder - it is called from two sites (`qwen4exp.cpp:589` in `can_reuse` and `:681` in the build) and gating only one desynchronises the reuse predicate from the built graph. It kills the pp regression *and* the 34 decode re-reserves. It gives up the prefill win, which at depth 24k is a 24x reduction in derived rows - a real win you have not yet disproved.

**(b) is the right shape, but you have scoped it too large.** You do not need the node set independent of run state in general. You need two things:

**(d1) Collapse REBUILD into CACHED-with-`wm=0`.** Today they differ only in *where the scores read from*: REBUILD scores the freshly derived contiguous `pooled` (`qwen4exp.cpp:737-770`) and writes a `view_4d` of it via `set_rows` (`:809-815`); CACHED scores a `get_pool` view (`:741-743`) and writes a separately derived chain (`:783-806`). The reason REBUILD must exist is an aliasing hazard: in CACHED, `set_rows(dst, ...)` writes into a view of the same tensor the `mul_mat` reads, with **no graph edge between them** - only node order saves you (`ggml_build_forward_expand` at `:815` happens to precede the `mul_mat` at `:828`). `ggml_set_rows` returns `ggml_view_tensor(ctx, a)` (`ggml/src/ggml.c:4018-4020`), so **feeding the scores from the set_rows result** creates a real dependency: `get_pool -> set_rows -> reshape_3d -> mul_mat`. Then wm=0 (REBUILD) and wm>0 (CACHED) are the *same* topology differing only in `n_new`, i.e. only in sizes. This deletes the duplicate derive chain - it is a simplification, not an addition.

**(d2) Make the reserve measure that topology.** This is the part your code already knows how to do: `src/llama-memory-hybrid-idx.cpp:1015-1016` carries the comment *"graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set / without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode"* - you already fixed this exact bug for `n_stream`. Do the same for the pool: when the context is the full/reserve context (`is_update == false`, `src/llama-memory-hybrid-idx.h:226`; the only callers are `llama-context.cpp:614`, `:839`, `:4029`, all measurement), have `qsa_pool_get` return the pooled variant with `wm = 0`, `n_bid = ceil(n_kv/ratio)`, `n_new = n_bid`. Then the measure graph has the runtime node/leaf set and every write-path tensor is sized at its maximum (`new_cells` `[4*6144,1]` = 98 KiB, `memb` `[128,24576]` f32 = 12.6 MiB), so **zero re-reserves**, exactly as pool-off/tensor already demonstrates.

(d1)+(d2) keeps both wins, touches ~40 lines in two files you already own, adds no ggml change, and removes a fragile ordering assumption. Re-run the golden-corpus gate: REBUILD's scores would now come back through the pool instead of the fresh tensor, which should be bit-identical (E044 already validated the pool as such) but must be shown, not assumed.

**Option you did not list, and should consider first:** do nothing to the code until you have the churn-free number. If rep 2 (which I argue below is churn-free) shows the pool *losing* at depth, (a) is the correct fix and (d) is wasted work.

---

## Your four questions

**1. Why are the measure numbers identical?** Because `qsa_pool_get` never gets to run its predicate at reserve time. It reads the **live** indexer cells (`src/llama-memory-hybrid-idx.cpp:344`, `mem_idx->get_cells(...)`) and bails at `:360` on `nu == 0`, which is always true during `sched_reserve()` - the cache is empty in the constructor and after every `llama_memory_clear`. `init_full()` supplies a worst-case `n_kv` and `n_stream` but cannot supply cells. So both arms build NONE. The `can_reuse` predicate at `qwen4exp.cpp:589-597` is comparing against a graph that was built from the same NONE answer, which is why it also can't rescue you.

**2. Node counts.** `sched->graph` is **not** the model graph. `ggml_backend_sched_split_graph` builds `graph_copy` from the splits, and for every cross-backend split input it emits **two** nodes - an `input_dep` view and the `input_cpy` tensor (`ggml/src/ggml-backend.cpp:1500-1516`), the latter being exactly the `Meta(ROCm0)#attn_inp_kq_mask#0` you saw (named at `:1406`). That also resolves your `leaf_118` confusion: your print loop iterates nodes, and the input copies *are* nodes of `graph_copy` even though the source is a leaf of the model graph. (`leaf_%d` naming is `ggml/src/ggml.c:7285`, for unnamed leaves.)

So `+50/+38` need not be multiples of 12: the 12 QSA layers share **one** `llm_graph_input_qsa` because they share ratio 4 (`qwen4exp.cpp:670-672`), so input-set deltas are x1, not x12. What *is* clean: **6825 - 6813 = 12 = exactly one node per QSA layer**, which is the signature of REBUILD (derive chain + `set_rows` whose src is a view -> +1 node/layer) vs CACHED. The absolute +50/+38 mixes per-layer model nodes with per-input copy pairs and cannot be decomposed from `sched->graph.n_nodes` alone. Fix the probe by printing `ggml_graph_n_nodes(gf)`/`n_leafs` (the model graph, in `process_ubatch` before `ggml_backend_sched_alloc_graph`) next to `sched->graph.n_nodes`, plus `pool.mode/n_new/n_bid` per ubatch.

**3. Three measure prints.** One `sched_reserve()` call, three `graph_reserve` calls: pp (`src/llama-context.cpp:638`), tg (`:658`), pp again (`:681`, "reserve again with pp graph to avoid ggml-alloc reallocations during inference"). So 1134.82 / 20.90 / 1134.82. It matters in your favour: the **last** reserve is the pp worst case, so the reservation standing at the start of the run is 1134.82 MiB with pp-shaped `size_max` values. That is what makes the first re-reserve so destructive - it replaces a worst-case budget with a first-ubatch budget.

**4. `-sm none` 24 re-reserves, no loss.** Not mirror width, and not the sync loop *as a cost*. No physical realloc can occur in either mode (C4), so a re-reserve is: full device drain + `reserve_n` simulation + a second `alloc_graph` pass ~= 16 ms of real work + whatever device time was in flight. `-sm none` shows the expected result: **24 re-reserves move ~24 x (in-flight time) from `graph:compute` into `graph:alloc` and change the total by nothing** (699.08 vs 701.09 is inside your 1% noise floor). That is the strongest single piece of evidence in your whole dataset, and it argues *against* your C4 rather than for it. What I could not explain by reading is why `-sm none` pool-off triggers at all when `-sm tensor` pool-off triggers zero times - same model graph, same `n_buffers == 2`, same worst-case reserve. That is the one thing the `-v` probe must answer.

---

## Minimum decisive experiment (in cost order)

**1. Free - you already have the output.** Two numbers from the runs you ran:
- The `[prof] exit` table (the *timed* passes). You quoted `[prof] warmup`. If the exit table shows `sched:realloc_*` with 8 calls and a much smaller ms/call, the timed population behaves differently from the warmup and C6 is settled without any new run.
- The `+/-` column of the `pp8192` row (`llama-bench.cpp:2077` prints `avg +/- stdev`), or `samples_ns` via `-o json` (`:1810`). **I predict the pooled arm is bimodal: rep 1 ~= 520 t/s, rep 2 ~= 980 t/s, stdev ~= 300.** Reason: with `--no-warmup -r 2`, rep 1 runs 16 depth ubatches + 8 timed ubatches, each growing -> 24 re-reserves (matches your count exactly); rep 2 restores the cached depth-16384 state and every tensor now fits the budget rep 1 grew -> **0 re-reserves**. If instead the stdev is ~10, both reps churn or neither does, and my whole ratchet model is wrong. This one number decides whether the pp regression is steady-state or a rep-1 artifact.

**2. ~2 minutes of GPU - add `-v`.** Your build has `NDEBUG` off (`build/ggml/src/CMakeFiles/ggml-base.dir/flags.make`: `-g -O2 -g3 -fno-omit-frame-pointer`, no `-DNDEBUG`), and `llama-bench -v` skips `llama_log_set(llama_null_log_callback)` (`llama-bench.cpp:2258-2260`), routing ggml's unfiltered default logger to stderr (`src/llama-impl.cpp:32-36`, `ggml/src/ggml.c:314-319`). You then get, for every re-reserve, the exact trigger:
- `ggml-alloc.c:1012` "graph has different number of nodes" / `:1019` leafs -> topology
- `:1030` "node X is not valid" / `:1041` "src N (Y) of node X is not valid" -> **names the tensor** whose `size_max` was exceeded
- `:929` "reallocating %s buffer from size X to Y MiB" -> **if this line is absent, C4 is dead**, which is my prediction
- `ggml-backend.cpp:1678` "failed to allocate graph, reserving (backend_ids_changed = N)"

Shrink it: `-p 2048 -d 4096 -ub 1024 -r 1 --no-warmup` for both `Q4EXP_POOLED=0/1` and both `-sm none/tensor`. Four runs, ~8 minutes total, and it answers C2, C3, C4, C5, Q2 and Q4 with named tensors instead of inference.

**3. ~10 minutes - the churn-free A/B you actually wanted.** Same command with `-o json -r 3`, compare **rep 3 only** between arms. That isolates the pool's device-side effect at depth with zero reservation ramp. This is the number that should decide C7, and it is the only measurement in this whole thread that can.

If you want attribution rather than a decision, the 4-line addition is two nested regions inside the re-reserve branch at `ggml-backend.cpp:1693` - one around the `for n_backends: synchronize` loop, one around `reserve_n` + the second `alloc_graph`. That splits 512.5 ms into "device wait" vs "host work" definitively and ends the C6 argument.

---

## Residual risks and gaps I could not close by reading

- **Why `-sm none` pool-off triggers 24 re-reserves.** No code-level explanation found; the probe in (2) names it. Until then, the none/tensor contrast is an unexplained observation, not evidence.
- **Whether the HIP enqueue throttles.** My `D_on ~= 1350 ms` inference depends on `graph:compute(off) ~= D_off`, which holds if launches block on a full queue and fails if the pp graph is captured/replayed. E008 established capture for tg; I did not verify it for a 6825-node pp graph. If it *is* captured, `compute(off)` is mostly the input-copy drain and the arithmetic changes (though the sum argument, and therefore the +11.3% warmup conclusion, does not).
- **Your instrumentation has a real hazard.** `ggml_gallocr_reserve_n_size` at `ggml-backend.cpp:1636` runs *before* the sync loop at `:1693`, and it is not a pure query: `reserve_n_impl(no_alloc=true)` calls `ggml_vbuffer_free(galloc->buffers[i])` whenever `realloc` is true (`ggml-alloc.c:935-937`). If a re-reserve ever *does* need a bigger buffer, you are freeing a HIP device buffer with work in flight. It did not bite here (no realloc fires, per C4), but it would bite on the first genuine growth, and it makes the instrumented build unsafe as a correctness reference.
- **`prev` in the diff print is a function-static** (`ggml-backend.cpp:1626-1627`) that persists across arms within a process and across the pp/tg transitions, so the "X -> Y MiB" columns can compare tensors from different graphs. Your `leaf_118` entry is likely a casualty of this.
- I did not run anything on the GPU. Every claim above is from code plus your reported numbers.

**Blunt summary:** the causal story is half right. The topology mismatch (C2) and the size ratchet (C5-premise) are real and correctly identified, and the fix follows from them. But the *cost* model (C4) is wrong - nothing is being reallocated, and the 575 ms is your own previous ubatch's device time wearing a different label - which means the `-29%` (C6) is the same money counted twice, and the pp regression you are trying to explain may be mostly a rep-1 reservation ramp that a churn-free rep would not show. Check the stdev before you write a single line of the fix.
