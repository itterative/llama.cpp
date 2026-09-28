# E072 - the prefill realloc ratchet is the MoE weighted-reduction fusion's alloc-dep

Status: **closed**. Root cause found and the ratchet is removable; the fix is an inconsistency in how the
fusion reports its dependencies.
Tier T1, dev-rx9070-16g, 2026-09-28, ROCm 10.0.0. Parents: E069, E071.

## Prediction (stated before the run)

E071 localised the structural difference to one dependency node, and a code search put its origin in the HIP
MoE weighted-reduction fusion's `add_alloc_dep` (`ggml-cuda.cu:4597`). So: with `GGML_CUDA_DISABLE_FUSION=1`
no graph should carry that dep, the reserve graph and the runtime graph should have the same node count, and
the reallocs should fall to zero.

## Result

| arm | reallocs | `realloc_size` | prefill wall | peak VRAM | PPL |
| --- | --- | --- | --- | --- | --- |
| control | 34 | 1689-1776 ms | 5539-5725 ms | 7836-7838 MiB | 263100.7437 |
| `GGML_CUDA_DISABLE_FUSION=1` | **0** | **0** (region never fires) | **5144.78 ms** | 7838 MiB | 263100.9174 |
| + `LLAMA_RESERVE_WORST_CASE=1` | **0** | **0** | 5016.90 ms | 8369 MiB | - |

Both fusion-off arms also report **0 tensor trips and 0 structural trips**. Prediction confirmed. The width
bump is not needed for it (the arm without it is already clean) and only costs +530 MiB, so it is reverted
again.

## The chain

1. `ggml_backend_cuda_graph_optimize` matches `(experts * expert_scale) * router_weight` or
   `experts * router_weight` and reports `add_alloc_dep(experts, dst)` and `add_alloc_dep(weights, dst)`
   (`ggml-cuda.cu:4582-4603`).
2. `ggml_backend_sched_split_graph` turns each reported dependency into a `ggml_view_tensor` node in the
   graph copy (`ggml-backend.cpp:1527-1536`): "add a dependency node so that the kept tensors are not freed
   before this node is computed".
3. The reserve graphs carry one such node for `ffn_moe_down-3`; the prompt graphs do not (E071: 703 vs 702
   nodes, differing by exactly that node).
4. `ggml_gallocr_needs_realloc` tests the node count before any size, so the first prompt ubatch retightens
   the whole reservation (E069), and every ubatch after it is one padding step past the record: 34
   re-reserves, 1699 ms of a 5699 ms wall.

Minimal corner: the probe shows the reserve graphs match the fusion with `ne = 2560,10,1`, and in the prompt
phase layers 0-2 match while layer 3 does not - yet the prompt graph carries no dep either. Which per-graph
condition decides the dep's presence is still open; the experiment above does not depend on it.

## Why this stayed invisible for three experiments

The dep node is bookkeeping, not arithmetic. It never appears in the per-node assignment dump
(`print_assignments` skips view ops), it changes no number, and it does not show up in a size-based search:
it only moves the node count, which is the one thing that decides whether a reservation survives.

## The dep is self-defeating during prefill

`ggml_cuda_check_fusion_memory_ranges` (`ggml-cuda.cu:3049`) re-checks aliasing at compute time and skips the
fusion when the input and output ranges overlap. The dep exists so that they do not overlap - so a graph
without the dep is a graph where the fusion can be skipped. That is consistent with this arm *improving*: the
fused compute it gives up is worth less than the 1.7 s the ratchet costs. Open: count how often the range
check actually skips the fusion during prefill (needs a counter in the check).

## Numerics

Fusion-off PPL is `263100.9174` against `263100.7437` - a summation-order change, the same class as the mmvf
unroll move, because the fusion reduces several mul/add pairs in one kernel. Explained, but a fusion-off arm
now has its own golden value.

## Fix directions

1. Make the dep reporting consistent per graph: a graph whose nodes will actually be fused should carry the
   dep, and a graph whose nodes will not, should not. The current asymmetry - dep in the reserve graph,
   absent at runtime - is the only reason a worst-case reservation cannot survive, so fixing it removes the
   ratchet and keeps the fusion.
2. Keep the fusion off for prefill-shaped graphs. Cheap and measurable, but it gives up whatever the fusion
   is worth in decode (not measured here).

## Commands

```sh
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib GGML_CUDA_DISABLE_FUSION=1 GGML_PROF_REGIONS=1 \
  build/bin/llama-cli -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 -n 1 -f $F
# the fusion probe needs -v and GGML_ALLOC_DEBUG_REALLOC=1
```