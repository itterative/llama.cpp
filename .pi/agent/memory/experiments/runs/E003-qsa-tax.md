# E003 - what does the QSA machinery cost when its sparsity is not collected

- date: -
- machine: dev-rx9070-16g (ablation, T1) -> bench-4x-r9700-32g for the real effect size
- tier: T1
- status: planned
- parent: E002
- commit: -
- build: as E002
- model: synthetic qwen4exp, context swept

## hypothesis

Because `build_attn_mha` is called with `n_kv_max = 0` while the indexer and the mask
rebuild are still executed (`src/models/qwen4exp.cpp:766-767`), qwen4exp pays a
context-scaling tax for sparsity it does not get. That tax is large enough on RDNA4 to
justify porting the mask compaction to HIP (backlog H4b).

## prediction

Deciding metric: `pp` t/s on the synthetic model, swept over n_ctx, comparing three
variants at each point.

| variant | what it is |
|---|---|
| V0 | current tree (indexer + mask rebuild, dense compute) |
| V1 | indexer + mask rebuild removed, plain dense `kq_mask` - **semantics change**, so this is an upper bound on the saving, not a legal optimisation |
| V2 | indexer kept, mask rebuild removed by reusing one cached all-visible mask - semantics change again, but isolates the `fill(-INF)` + `set_rows` + `add` traffic alone |

Expected: V1-V0 gap grows roughly linearly in context length, and at the largest context
the synthetic model can reach the gap exceeds 10%. V2 isolates how much of that is mask
traffic vs indexer arithmetic.

Falsified if V1-V0 stays under 2% across the sweep. That would mean the tax is not the
problem and the effort belongs on H2 (HC fusion) and the MoE path instead - a genuinely
useful negative, because it redirects the whole project.

## conditions

To be filled at run time from `PROTOCOL.md` 5.1. The ablations are patches, so each
variant needs its own sha-or-patch recorded in `results/E003-<variant>.patch`.

## known limitations, stated up front

- A synthetic model has the right graph shape and the wrong dimensions: `indexer_top_k`,
  `compress_ratio`, `n_full_attention_layers` and n_embd_head all differ from the real
  Qwen3.8-Flash-Next config, so the *absolute* percentage here is not the real one. This
  experiment is about whether the tax exists and how it scales, not its magnitude.
- The real magnitude requires T2 on the bench box at long context, and the model dims,
  which the user must supply (backlog P5).
- V1/V2 change outputs. They are instruments, not proposals. If either "wins", the number
  is still a ceiling.

## results

| variant | n_ctx | pp median | min-max | vs V0 | n |
|---|---|---|---|---|---|
| (pending) | | | | | |

## raw

`results/E003-*.patch`, `results/E003-llama-bench.csv`.

## verdict

pending

## notes

Sequence: E002 must land first (a stable synthetic baseline), and B1 (the
`test-backend-ops` crash) matters here because V1/V2 alter numerics - without a working
op gate the only available check is that V0 output is unchanged, which is not a check on
the ablations at all.
