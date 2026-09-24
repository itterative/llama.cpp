# E050 - 4-card confirmation: the gate removes the pp regression, and pooling is a loss below ~32k

Raw: [results/user/h9-postfix-a8ec1040b/](../results/user/h9-postfix-a8ec1040b/) (`llama-bench-nopool.log`,
`llama-bench-pool.log`, both with `GGML_PROF_REGIONS=1`), build `06c3af843`, real Q4_K_M 176.94 B /
111.38 GiB, 4x R9700, `-sm tensor -fa 1 -lm none -lzm on-direct -ot per_layer_token_embd=CPU -b 2048
-ub 1024 -r 3`. Same session, both arms, so this satisfies non-negotiable 2, which E045 could not.

## t/s

| test | pool off | pool on (gated) | delta | pre-gate pool (E045) |
| --- | --- | --- | --- | --- |
| pp8192 @ d4096 | 2266.30 | 2246.95 | -0.9% | 1766.27 (-21.9% vs that baseline) |
| pp8192 @ d16384 | 2150.43 | 2138.89 | -0.5% | 1713.25 |
| pp8192 @ d40960 | 1969.44 | 1959.43 | -0.5% | 1594.10 |
| pp8192 @ d131072 | not captured | **1508.89** | - | 1260.05 |
| pp4096 @ d131072 | not captured | 1468.74 | - | 1250.89 |
| tg128 @ d4096 | 34.18 | 32.44 | **-5.1%** | 32.39 |
| tg128 @ d16384 | 32.79 | 31.98 | **-2.5%** | 32.28 |
| tg128 @ d40960 | 30.83 | 31.64 | +2.6% | 31.84 |
| tg128 @ d131072 | not captured | **30.12** | - | 30.14 |

The d131072 pool-off rows are missing (that arm ends at d40960 in the captured log), so the deep
comparisons lean on E045's numbers from a different build and session - a data point, not a matched
pair. Both are needed if the d131072 claim has to hold up: 1508.89 vs E045's 1502.33 baseline is
+0.4% where the pre-gate pooled arm measured 1260.05, and 30.12 vs 30.14 pre-gate vs 23.16 pre-H9 says
the decode win is untouched by the gate.

## Profiler evidence

- `sched:realloc_buft`: **0 calls in most tables, 6 calls @ ~24.7 ms in the deepest warmup**. On one
  card the gated build showed the same 6 @ 23-25 ms, so four mirrors and 111 GB of weights change
  nothing - E049's gate holds at scale.
- `measure graph: nodes = 6775, leafs = 1358` identical across all depths and both arms; the need grows
  with the context as expected (1073 MiB @ d4096, 1135 @ d16384, 2241 @ d131072). The mismatch that
  caused the regression is a node-count difference against the *runtime* graph, not a size problem.
- Steady-state decode tables look like this: `graph:compute 385 calls 8.44 ms/call`,
  `graph:alloc 7 calls 25.96`, `graph:set_inputs 385 calls 2.21`, `meta:subgraph 37345 calls 0.028`,
  `meta:allreduce 36960 calls 0.015` - i.e. per decode token ~8.4 ms of compute region including device
  wait, ~0.5 ms of allreduce across 96 subgraph dispatches.
- Prefill tables at depth show `graph:set_inputs` at **~100-124 ms per 1024-token ubatch**, second only
  to `graph:compute` (~266-444 ms), and it is identical in both arms. Worth a look at some point:
  ~110 ms of the ~300 ms ubatch is spent before the device starts.

## What this changes operationally

Pooling is now a pure-decode feature that only pays off on long contexts: **negative below ~32k, +2.6%
at 40k, ~+30% at 131k**, with prefill untouched (-0.5..-0.9%). Since `Q4EXP_POOLED` is an env var, the
right guidance for now is to leave it off unless the working context is deep, rather than adding a
depth threshold in the code - a threshold would reintroduce exactly the runtime topology change that
caused E049, trading 25 ms re-reserves at the crossing point for a per-depth gain that is small in the
band where it flips.

The shallow negative is inherent, not the gate's doing: with few blocks there is nothing worth pooling,
and the graph still carries the pool read tensors. E045's pre-gate rows showed the same sign
(32.39 vs 33.65 at d4096).

## Loose ends

1. Missing d131072 pool-off pair (see above).
2. GPU hangs during these runs, reported by the user as "not the point right now" - unresolved, and
   E045's hang analysis (pool VRAM at ctx 245760, `common_fit_params` refusing under tensor split)
   still stands as untested context.
3. `graph:set_inputs` ~110 ms/ubatch in prefill, unexamined.
4. E049's open question is still open: whether pooling prefill can win at all when measured without the
   reservation confound (the reviewer's d1+d2 route).
</content>
