# E066 - what changes the graph node count during a cli prompt

- date: 2026-09-28
- machine: dev-rx9070-16g (1x gfx1201), ROCm 10.0.0
- tier: T1
- status: running
- parent: E065 (found the per-ubatch realloc and its forced sync), H19 (leaves the node-count changes unnamed)
- raw: [results/E066-node-count-probe/](../results/E066-node-count-probe/)

## question and what decides it

H19 names the *size* trips (`blk_cells` and the attention inputs, n_kv-proportional) but explicitly leaves
its first act unnamed: "What are the two node-count changes?". E065 adds a datum that does not fit the
256-padding story - during a prompt it fires ~1.6 `sched:realloc_size` events per ubatch (34 for 21
ubatches) where decode fires 1 per 256 tokens - so the prompt's ratchet needs its own explanation.

Deciding metric: the per-ubatch tuple `(n_tokens, n_seqs, n_outputs, nodes)` from a probe at the end of
`process_ubatch`, against the `sched:realloc_size` call count for the same run. `n_leafs` is left out on
purpose: the cgraph struct is internal to ggml (`ggml/src/ggml-impl.h:341`), so the probe uses the public
`ggml_graph_n_nodes()` and does not reach into ggml internals for the field.

Note on the background datum: E065 reported "~1.6 events per ubatch" for the prompt phase. That was a
denominator error on my side - 34 events were divided by decode *calls* (21), not ubatches (36), and the
`-ub 128` arm below settles the rate directly.

Hypothesis, written before running: the node-count changes are the **output/fused-sampler** family. The
context already keeps two graphs, `gf_res_prev[n_outputs > 0]`, so the count should step exactly where
`n_outputs` flips and stay constant across ubatches that share it.

Falsified if `nodes` varies between ubatches with identical `n_outputs`. That would mean the topology itself
is n_kv-dependent (the QSA/cell inputs adding and removing nodes), which is a different fix target than
"keep the reservation constant across a prompt".

Not in scope: the *size* trips. Node counts cannot see them; `GGML_SCHED_DEBUG=2`'s assignment listing is
the tool for those and H19 already used it.

## arms

| arm | invocation | expected prints |
| --- | --- | --- |
| pp-ub1024 | sparse corpus, `-c 32768 -b 2048 -ub 1024 -n 1` | 21 prompt ubatches + 1 decode |
| pp-ub128 | same, `-ub 128` | ~255 prompt ubatches (does the event count follow ubatches or n_kv steps?) |
| tg | same prompt, `-n 256` | 94 decode steps, constant expected |

Env: `LLAMA_UBATCH_DEBUG=1 GGML_PROF_REGIONS=1`, `LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib`. The probe
prints only, so numerics are untouched and no correctness gate applies; the golden PPL on this toolchain is
263100.7437.

## commands

[`results/E066-node-count-probe/commands.sh`](../results/E066-node-count-probe/commands.sh)

## results

Probe lines equal `process_ubatch` invocations: 36 (pp-ub1024), 259 (pp-ub128), 130 (tg-ub1024).

**1. The inference graph's node count is constant: `nodes = 667` in every line of all three arms** - at 1,
2, 4, 51, 128, 874 and 1024 tokens, with `n_outputs` both 0 and 1. The hypothesis is falsified in the strict
sense: the count follows neither `n_outputs` nor `n_kv`. H19's "two node-count changes" are therefore not
changes of *this* graph - the reserve path (`gf_res_reserve`, built from its own `ubatch_reserve` shape) is
the remaining candidate, and that is a probe for a later run, not a blocker.

**2. The realloc rate is one event per 256-token padding crossing, and the `-ub` sweep proves it:**

| arm | ubatches | `sched:realloc_size` calls | rate |
| --- | --- | --- | --- |
| pp-ub1024 | 36 | 34 | 1 per 961 tokens (~1 per ubatch) |
| pp-ub128 | 259 | 131 | **1 per 253 tokens** |
| tg-ub1024 | 130 | 34 | as pp-ub1024; the 94 decode steps add none |

With `-ub 128` each ubatch is half a padding step, so the per-ubatch rate halves while the per-token rate
stays pinned to 256; with `-ub 1024` every ubatch crosses at least one boundary, so it is one per ubatch.
That is exactly H19's `get_n_kv` 256-padding, and it retires E065's stray datum: 34 events over 36 ubatches
is ~0.94 per ubatch, not 1.6. The padding story does explain the prompt phase; no extra mechanism is needed.

**3. No GPU hangs or faults** in any arm. The lines matching `error` are `common_params_fit_impl: will leave
9382 >= 1024 MiB of free device memory, no changes needed`, `slot reset` and
`ggml_backend_cuda_graph_compute: CUDA graph warmup reset` - all benign.

Method note: the probe is a `LLAMA_LOG_INFO` line and this build's default level is WARN, so the first
attempt produced zero output despite a successful build and a clean run. `-v` is required - the same trap
`experiment-protocol.md` records for the pool mode line.

## reads against

- **H19's fix target is sharp and its blocker is gone:** the trips are size-only, the node count never moves,
  and the rate is the 256-token padding boundary. "Keep the buffer sizes constant inside a reserved window"
  (H19's own conclusion - preserve the worst-case budget through the prompt) is the whole fix.
- The prompt-phase instance is the *same* mechanism as the decode one, hit once per ubatch whenever the
  ubatch is wider than 256 tokens. One fix, three instances (H11, H19, H22).
- E065's prefill numbers stand (34 reallocs, 1689 ms, 97% of it the forced wait), with the rate corrected to
  ~1 per ubatch / 1 per 256 tokens.
- Still open if ever needed: the reserve path's graph shape. Not on the critical path for the fix.