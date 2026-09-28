# E073 - fix: report fusion dependencies independent of batch work

Status: **closed, fixed**. The prefill ratchet (H19) is gone with the fusion still enabled.
Tier T1, dev-rx9070-16g, 2026-09-28, ROCm 10.0.0. Parents: E069, E072.

## Prediction (stated before the run)

Make the alloc-dep reporting work-independent, keep the fusion: expect 0 reallocs, no new VRAM, and a
**bit-identical** PPL, because only dependency reporting changes, not arithmetic.

## Result

| arm | reallocs | `realloc_size` | prefill wall | peak VRAM | PPL |
| --- | --- | --- | --- | --- | --- |
| control | 34 | 1689-1776 ms | 5539-5725 ms | 7836-7838 MiB | 263100.7437 |
| fusion off (E072) | 0 | 0 | 5144.78 ms | 7838 MiB | 263100.9174 |
| **fix, fusion on** | **0** | **0** (region never fires) | **4818.84 / 5075.23 ms** | 7838 MiB | **263100.7437** |

All three parts of the prediction hold. The fix keeps the fusion's compute and drops the ratchet, so it is
faster than turning the fusion off, and it costs no VRAM - unlike the E068 width-bump route (+530 MiB).

## The failing test

`ggml_cuda_match_moe_weighted_reduction` ends its shape checks with

```c
const int64_t n_tokens = weighted->ne[2] * weighted->ne[3];
if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
    return false;
}
```

During prefill the last layer's reduction has `ne[2] = 0`: the batch carries no output tokens, so that
layer's MoE output is not needed. The probe shows it directly - matched with `ffn_moe_down-3 ne=2560,10,1`,
rejected with `ne=2560,10,0`. Not fusing an empty reduction is right; losing the *dependency* with it is
not, because the scheduler turns dependencies into keep-alive view nodes and the graph's node count is what
decides whether a reservation survives. Measured over one run: layers 0-2 reported dependencies in all 47
graphs, layer 3 in only 14 - so 33 graphs had 3 dep nodes (702 nodes total) and 14 had 4 (703), and the
reservation was retightened on the first prefill ubatch of every batch.

## The fix

`ggml_cuda_match_moe_weighted_reduction` takes `require_work` (default true). The dependency-reporting call
site passes false, so an empty reduction still reports its dependencies; the compute call site keeps the
default, so an empty reduction is still not fused. Four lines: one parameter, one guard, one call site.

The invariant it restores, in one sentence: **a fusion must not change the graph's node count per batch.**
Fusions are per-graph heuristics, the scheduler's per-slot bookkeeping is indexed by node position, and the
node count is tested before any size - so any fusion whose applicability depends on the batch makes the
allocation depend on the batch, and a graph whose tensors grow step by step then re-reserves on every step.

This is the same family as E049/E056 ("the reservation must measure the shape the runtime builds"), one
level up: there the shape was a memory context kind, here it is a fusion verdict.

The fusion itself is upstream (`3466812d1 cuda: fuse MoE weighted expert reduction (#25952)`); this change is
a candidate to send upstream, and it is four lines, but it belongs to whoever owns the discussion.

## Commands

```sh
F=.pi/agent/memory/experiments/tools/sparse-corpus.md
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib GGML_PROF_REGIONS=1 \
  build/bin/llama-cli -m models/q4exp-4l.gguf -fa 1 -ngl 99 -st --temp 0 -b 2048 -c 32768 -ub 1024 -n 1 -f $F
# gate: PPL = 263100.7437, unchanged, because no arithmetic moved
LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib build/bin/llama-perplexity -m models/q4exp-4l.gguf \
  -f .pi/agent/memory/experiments/tools/golden-corpus.md
```