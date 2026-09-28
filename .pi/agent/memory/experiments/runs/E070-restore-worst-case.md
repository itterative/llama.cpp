# E070 - does re-establishing the worst-case reservation after the warmup remove the ratchet

Status: **running** (hypothesis, metric and prediction written before the first arm).

Tier T1, dev-rx9070-16g, 2026-09-28, ROCm 10.0.0, `models/q4exp-4l.gguf`. Parent: E069.

## Hypothesis

E069 showed the ratchet is not an under-sized reservation. `sched_reserve` asks for a worst-case graph
(703 nodes, `need = 1381.18 MiB`), then the warmup's structurally different graph (702 nodes) makes
`ggml_gallocr_reserve_n_impl` retighten every recorded size to the warmup's, and each prefill ubatch is
exactly one padding step past that record. So: reserve the worst case **again, after the warmup**, and the
record should cover the whole prompt.

Test hook, harness-only and gated: `LLAMA_RESERVE_RESTORE=1` makes `common_init_from_params` call
`llama_graph_reserve(lctx, n_ubatch, 1, n_ubatch)` once the warmup is done. That entry point reserves
through `memory->init_full()` - the whole cache, not the cells at that moment - which is the shape E069
identified as the missing piece.

| | prediction |
| --- | --- |
| `sched:realloc_size` calls | 34 -> 0-2 |
| `sched:realloc_size` ms | 1699 -> under 100 |
| `phase:prefill` wall | 5699 ms -> about 4100 ms, i.e. the time the realloc branch spends |
| peak VRAM | unchanged near 7838 MiB - the reservation already exists, so this adds no memory |
| PPL | bit-identical, 263100.7437 |

Falsifier: reallocs stay near 34, or the wall does not move. That would mean the record is not the whole
story - e.g. the reserve must be the *last* one before the prompt at the same node count, or something
below the record also changes per ubatch.

## Results

| arm | reallocs | `realloc_size` | prefill wall | tensor trips | structural trips | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| control | 34 | 1689-1776 ms | 5539-5725 ms | 32 | 2 | 7836-7838 MiB |
| restore, `n_outputs = n_ubatch` | 34 | 1708-1932 ms | 5475-5669 ms | 32 | 2 | 8280-8282 MiB |
| restore, `n_outputs = 1` | 34 | 1693-1698 ms | 5477-5480 ms | 32 | 2 | 7836 MiB |

**Falsified.** The extra reserve runs (the fourth `measure graph` line, and +442 MiB in the first variant), and the
ratchet is untouched.

## Why, and what it means

A reserve through the public entry point is **always 703 nodes**, whichever `n_outputs` is passed:
`graph_reserve` calls `ubatch_prepare_reserve` (`src/llama-context.cpp:2528`), which builds the all-outputs
convention (it selects one output token per sequence only for sequences that have a sampler, and a reserve has
none). The prompt pass runs the other convention - one sampled output token - and that graph has 702 nodes.

```
restore arm: reserve sequence (all 703 nodes)
  691.02 MiB / 26.43 MiB / 691.02 MiB   (sched_reserve)
  691.02 MiB                            (this hook, n_outputs = 1)
  0.02.339: graph structure changed, nodes 703 -> 702   (first prompt ubatch)
  0.07.759: graph structure changed, nodes 702 -> 703   (after the prompt)
```

`ggml_gallocr_needs_realloc` tests the node count **first**, before any size, and the per-slot bookkeeping is
indexed by node position, so a structural difference has to retighten the record whatever the sizes say. That
is the whole reason this hypothesis cannot work from the harness: there is no way to reserve the graph the
prompt will actually run.

So the lever is narrower than E069 hoped, and it is now precise: **the reserve must be made with the same
output convention as the batch that will run**. Either llama.cpp reserves with the runtime convention (the
sampler-aware output selection, not the all-outputs one), or the last reserve before a batch is the batch's
own shape. Both are llama.cpp-side, both need the reserve to also see a full-cache memory context, and both
are testable: the arms above are the control for them.

Correction to E069: the 702-node early graph is the first **prompt** ubatch, not the warmup - the warmup, like
every reserve, builds 703. Everything else in E069 stands: the retighten is the ratchet's cause and `qsa_bias`
is the tensor that shows it.

## Commands

```sh
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
for arm in control restore; do
  E=""; [ $arm = restore ] && E=LLAMA_RESERVE_RESTORE=1
  env $E LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib GGML_ALLOC_DEBUG_REALLOC=1 GGML_PROF_REGIONS=1 \
    build/bin/llama-cli -v -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 \
    -n 1 -f $F
done
```