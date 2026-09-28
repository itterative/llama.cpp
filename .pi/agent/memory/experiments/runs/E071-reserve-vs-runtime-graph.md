# E071 - why the reserve graph has 703 nodes and the runtime graph 702

Status: **closed**. Negative for E070's narrowed lever, and it localises the difference.
Parent: E070.

## Hypothesis

E070 concluded that the reserve must use the batch's output convention. If that were the whole story, a reserve
with `n_outputs = 1` would build 702 nodes like the prompt pass. E070's own log already contradicted it (all four
reserves were 703, including the `n_outputs = 1` one), so this run asked the question directly instead of
building the gate.

## What was done

`GGML_ALLOC_DEBUG_REALLOC=1` now also dumps the node name list once for the reserve graph and once for the first
runtime graph that trips the structure check. One control run, nothing else changed.

## Finding

The two graphs are identical except for **one scheduler dependency node**:

```
reserve, n669:  ffn_moe_out-3
                ffn_moe_down-3 (view)     <- only in the reserve graph
                ffn_gate-3
runtime, n669:  ffn_moe_out-3
                ffn_gate-3
```

`ffn_moe_down-3 (view)` is a `ggml_view_tensor` dependency the scheduler inserts for a tensor that is an input to
another split ("add a dependency to the input source so that it is not freed before the copy is done",
`ggml-backend.cpp`). So in the reserve copy graph the last layer's MoE down output is a cross-split input, and in
the runtime graph it is not. The node name sets are otherwise identical, and the count difference is that one node.

## Consequence

The output convention does not drive it: `n_outputs = 1` and `n_outputs = n_tokens` both give 703 nodes, measured.
So "reserve with the batch's output convention" cannot make the structures match - that lever is not viable as
scoped.

The remaining question is a graph-construction one and it is narrow: why does the reserve graph hand
`ffn_moe_down-3` to another split? Until that is answered, the node-count check keeps retightening the
reservation on the first prompt ubatch, and there is no safe allocator-side shortcut: the per-slot bookkeeping is
indexed by node position, so a graph with one node inserted shifts every later slot and the comparison would no
longer be slot-to-slot (E069's corruption result is the same hazard in a different disguise).

Which means H19's fix is blocked on a graph-identity question, not on a reserve-policy one.

## Commands

```sh
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib GGML_ALLOC_DEBUG_REALLOC=1 GGML_PROF_REGIONS=1 \
  build/bin/llama-cli -v -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 -n 1 \
  -f .pi/agent/memory/experiments/tools/sparse-corpus.md
# then diff the [reserve] and [runtime] node lists in the log
```