# E069 - what actually trips the prefill realloc ratchet

Status: **closed**. Supersedes E068 (whose "mask is covered, trips unchanged" reading was right about the mask
but wrong about the culprit: with `-fa 1` the attention mask is f16, and the f32 tensor in the probe is the
QSA per-block bias).

Tier T1, dev-rx9070-16g, 2026-09-28, `models/q4exp-4l.gguf`, ROCm 10.0.0.

Prompt: the user doubted the E068 target ("the realloc might not be part of the place you were trying") and
asked whether the QSA side can be fixed.

Hook/tooling: `GGML_ALLOC_DEBUG_REALLOC=1` (new, `ggml/src/ggml-alloc.c`) prints the tensor that fails the fit
test with its shape and the recorded size; the QSA graph inputs now carry names (`qsa_bias`, `qsa_blk_cells`,
...) so the printout is readable.

| | prediction (before the run) | observed |
| --- | --- | --- |
| culprit tensor | a QSA input, not the attention mask | **confirmed**: `qsa_bias` F32 `[n_blocks, n_tps, n_stream]` |
| fixing its reservation removes the trips | yes | **refuted**: it is already reserved at the full-cache shape |
| cost recoverable | ~1.7 s of the 5.7 s prefill wall | confirmed as a cost, not recoverable this way |

## What the culprit is

Every prefill ubatch fails on one tensor, `qsa_bias` (`src/models/qwen4exp.cpp:679`), sized from
`n_kv = mctx_hyb->get_idx()->get_n_kv()` (`:647`):

```
trip 2: src of qsa_bias (view): qsa_bias f32 ne=576x1024x1x1 need=2359296 have=1310720
trip 3: ... ne=832x1024x1x1 need=3407872 have=2359296
trip 4: ... ne=1088x1024x1x1 need=4456448 have=3407872
```

`have` is always exactly the previous ubatch's `need`. `n_blocks = ceil256(cells)/r` with `r = 4`, so each
1024-token ubatch adds 256 blocks and 1 MiB.

## Why it is not a QSA sizing bug

The reserve graph **already** builds the QSA inputs at the full cache: the 1-arg
`llama_kv_cache_context(llama_kv_cache *)` ctor (`src/llama-kv-cache.cpp:2706`) sets `n_kv = kv->get_size()`,
and `llama_memory_hybrid_idx` reserves through exactly that ctor for both halves. Measured with a temporary
`Q4EXP_QSA_DEBUG=1` probe in `build_qsa_top_k`:

```
qsa debug: r = 4, n_kv = 32768, n_blocks = 8192, n_tokens = 2048, n_tps = 2048, pool = 1     (reserve)
qsa debug: r = 4, n_kv = 1280,  n_blocks = 320,  n_tokens = 1024, n_tps = 1024, pool = 1     (first prefill ubatch)
```

So the sched_reserve pass asks for `[8192, 2048]` F32 = 67 MiB and `[32768, 2048]` F16 mask, and the prof line
records the reservation as `nodes = 703, leafs = 141, need = 1381.18 MiB`. **Nothing is under-sized.**

## The real mechanism: the reservation is retightened, not outgrown

Three facts, all measured:

1. The run makes exactly **3** `ggml_backend_sched_reserve` calls (all `need = 1381.18 MiB`), i.e. the prefill
   ubatches never reserve through `llama_context::graph_reserve`.
2. The run makes **34** reallocs, but only **32** tensor-level fit failures. With the new structural print the
   other two are:
   ```
   0.02.552: graph structure changed, nodes 703 -> 702, leafs 141 -> 141    (the warmup decode graph)
   0.08.197: graph structure changed, nodes 702 -> 703, leafs 141 -> 141    (back to prefill)
   ```
3. `ggml_gallocr_reserve_n_impl` **assigns** each slot's `size_max` from the graph it is reserving for:
   `node_alloc->dst.size_max = ggml_backend_buft_get_alloc_size(...)`. It does not keep the larger earlier value.

Chain: worst-case reservation (1381 MiB) -> the warmup decode graph has one node fewer -> structure check fails
-> the sched's fallback reserves for **that** graph -> every recorded size becomes the warmup's (the 1.3 MiB that
shows up as `have` at the first prefill trip) -> each prefill ubatch is bigger than the record by one padding
step -> one realloc plus one full device drain per ubatch.

Cost, control arm, `-b 2048 -ub 1024 -c 32768 -n 1`: `sched:realloc_size` **34 calls, 1699 ms** of a **5699 ms**
prefill wall, 7838 MiB peak.

## Two fixes that do not work, and why that is useful

- **`size_max` as a high-water mark** (keep the max, recompute the layout): arms abort with
  `GGML_ASSERT(i01 >= 0 && i01 < ne01)` in `ggml-cpu/ops.cpp:5015` - corrupt indices. `size_max` is a
  *placement-validity* token, not a capacity record: it says "this slot's address was laid out for a tensor of
  this size". Keeping a bigger value from an older layout lets a later graph run against a layout computed for
  other shapes. The buffer chunks do keep their capacity, but the placement does not survive.
- **Skip the reserve when the graph fits** (`GGML_ALLOC_KEEP_RESERVATION`): inert by construction. The reserve
  path is only ever entered *because* the fit test failed, so a fit test inside it cannot short-circuit
  anything. It fired 32 times in the gate-on arms and skipped nothing.

## Conclusion

The QSA side cannot be fixed from the QSA side. This is not a tensor that is under-reserved, it is a
**reservation that keeps being re-made for the wrong graph**, and the only lever is the policy that decides when
`ggml_gallocr_reserve_n` runs and what graph it measures. Directions, in order of how local they are:

1. Make the retighten measure a worst-case graph. The sched's fallback reserves with `sched->graph`, which is
   built from the *current* memory context; a graph built from a full-cache context (`init_full`, the same
   context `sched_reserve` uses) would keep the record at the worst case. Needs a way for llama.cpp to hand the
   sched a measure graph, or for the fallback to be told the worst case.
2. Keep the reservation across a structural change instead of recomputing it for the new structure - i.e. a
   persistent per-slot capacity record that is *also* correct about the placement. The negative result above is
   the guard rail: any version that only keeps sizes will corrupt.
3. Accept it, and treat the 1.70 s as the price of the current design. Note E056 already fixed this class of
   ratchet once for the pooled graphs by keying the pool's worst case on the context kind; the same shape of fix
   is needed here, one level up.

## Commands

```sh
# name the culprit (needs -v; the probe is opt-in)
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib GGML_ALLOC_DEBUG_REALLOC=1 GGML_PROF_REGIONS=1 \
  build/bin/llama-cli -v -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 -n 1 \
  -f .pi/agent/memory/experiments/tools/sparse-corpus.md

# correctness gate, bit-identical across every arm of this experiment
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib build/bin/llama-perplexity -m models/q4exp-4l.gguf \
  -f .pi/agent/memory/experiments/tools/golden-corpus.md     # PPL = 263100.7437
```

Kept from this experiment: the `GGML_ALLOC_DEBUG_REALLOC` probe and the QSA input names. The E068
`LLAMA_RESERVE_WORST_CASE` scaffold (width bump + `n_kv` floor) is reverted: it buys +530 MiB of VRAM, removes
zero reallocs, and the floor was redundant because the 1-arg ctor already reports `get_size()`.