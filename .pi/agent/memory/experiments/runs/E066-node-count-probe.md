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

Deciding metric: the per-ubatch tuple `(n_tokens, n_seqs, n_outputs, nodes, leafs)` from a probe at the end
of `process_ubatch`, against the `sched:realloc_size` call count for the same run.

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

(pending)

## reads against

(pending)