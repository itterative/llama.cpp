# INDEX - qwen4exp / RDNA4 experiments

Append-only ledger. One row per experiment, newest at the bottom (ids never reuse).

**Contract:** the `headline` cell is one clause - the verdict plus the one number that says whether to
read the record. It is for answering "did we already try that?" and "what happened?", nothing else. All
analysis lives in `runs/`; when a row starts carrying mechanism, caveats or a second result, it has grown
out of contract and the fix is to move that text into the record, not to widen the row.

Verdicts: `planned` (reserved, not run), `open` (ran, question still live), `done`, `dead-end`,
`not attributed`. Ids with no row here are still open in [plans/backlog.md](plans/backlog.md).

| id | date | tier | machine | hypothesis (one line) | headline | verdict | record |
|---|---|---|---|---|---|---|---|
| E001 | 2026-09-17 | T1 | dev-rx9070-16g | the T1 harness (dummy models + fusion counts + bench) runs on ROCm0 | harness works; `test-fusion` is impossible here (Metal-only API) | done | [runs/E001-t1-harness-viability.md](runs/E001-t1-harness-viability.md) |
| E002 | 2026-09-17 | T1 | dev-rx9070-16g | a synthetic qwen4exp model gives a usable pp/tg baseline | no: a 19 MB model measures harness overhead | dead-end | [runs/E002-synthetic-baseline.md](runs/E002-synthetic-baseline.md) |
| E003 | - | T1 | dev-rx9070-16g | qwen4exp pays a context-scaling QSA tax for sparsity it cannot collect on ROCm | blocked on E004; the `n_kv_max` flip is inert | planned | [runs/E003-qsa-tax.md](runs/E003-qsa-tax.md) |
| E004 | - | T1 | dev-rx9070-16g | a config-shaped synthetic model (real head_dim 256 / 24-2 heads / hc_lowrank 320 / budget 2048, fewer experts) is a usable pp instrument | - | planned | (record not yet written) |
| E005 | 2026-09-17 | **T2** | bench-4x-r9700-32g | real qwen4exp Q4_K_M on 4x R9700 gives us a baseline | first real numbers: tg 28.2 t/s, ~9% GPU util; not a usable baseline (fork build) | done | [runs/E005-real-baseline.md](runs/E005-real-baseline.md) |
| E006 | 2026-09-17 | T2 | bench-4x-r9700-32g | `-sm tensor` costs tg via per-layer cross-GPU collectives (no Infinity Fabric on Navi 48) | refuted by sign: layer split does ~zero collectives and is slower | done | [runs/E006-split-mode.md](runs/E006-split-mode.md) |
| E007 | - | T2 | bench-4x-r9700-32g | move the n-gram table onto VRAM to avoid the host gather | impossible: the table is mirrored, ~30 GB per card | dead-end | folded into [plans/ple-prefetch.md](plans/ple-prefetch.md) |
| E008 | 2026-09-17 | T2 | bench-4x-r9700-32g | the CPU-placed table might defeat HIP graph capture | no: capture is active, graphs off costs ~7% | done | [runs/E008-graph-capture.md](runs/E008-graph-capture.md) |
| E009 | 2026-09-17 | T2 | bench-4x-r9700-32g | the graph may be reallocated at unchanged size every step | no: same-size realloc ruled out | done | [runs/E009-graph-realloc.md](runs/E009-graph-realloc.md) |
| E010 | 2026-09-17 | T2 | bench-4x-r9700-32g | graph reuse may already be broken for this model | reuse works and is worth ~22 ms/step | done | [runs/E010-graph-reuse.md](runs/E010-graph-reuse.md) |
| E011 | 2026-09-17 | T2 | bench-4x-r9700-32g | if tg scales super-linearly with concurrency the cost is per-step host work | yes: ~28 ms/step is fixed, so ~90% of a decode token is per-step cost | done | [runs/E011-concurrency.md](runs/E011-concurrency.md) |
| E014 | - | T2 | bench-4x-r9700-32g | `-lzm off` + `-ot ...=CPU` gives a resident table with no demand paging; if tg improves, faults are part of the fixed ~28 ms | - (revived; `-ot` found inert, see the record's header) | planned | [runs/E011-concurrency.md](runs/E011-concurrency.md#found-in-the-header-of-this-log--ot-has-never-done-anything) |
| E015 | 2026-09-17 | T2 | bench-4x-r9700-32g | scheduler op-offload may be the fixed per-step cost | no: zero effect | dead-end | [runs/E015-no-op-offload.md](runs/E015-no-op-offload.md) |
| E017 | 2026-09-17 | T1 | dev-rx9070-16g | a shape-faithful dummy can be generated and run locally | yes: 5.47 ms/step, of which ~4.5 ms is per-step host work | done | [runs/E017-dummy-harness.md](runs/E017-dummy-harness.md) |
| E018 | 2026-09-17 | T1 | dev-rx9070-16g | a zero-filled dummy cannot detect a model-level bug | yes: fingerprint gate at PPL 262938.7619, noise floor ~1% | done | [runs/E018-correctness-gate.md](runs/E018-correctness-gate.md) |
| E019 | 2026-09-17 | T1 | dev-rx9070-16g | the op-level gate needs a baseline on the current stack | yes: 5633 OK / 7 FAIL, all pre-existing FA 192/128; `-j 1` only | done | [runs/E019-ops-baseline.md](runs/E019-ops-baseline.md) |
| E020 | 2026-09-17 | T1 | dev-rx9070-16g | can sparse FA be made to run on RDNA4 | yes: pp512 +23.3% @40k / +58.5% @164k; RDNA needed the 1x16 tiling | done | [runs/E020-sparse-fa-rdna4.md](runs/E020-sparse-fa-rdna4.md) |
| E021 | 2026-09-17 | T1 | dev-rx9070-16g | what actually owns the decode depth slope | the host-side `O(n_kv)` QSA scan: 13% of the step, ~2/3 of the slope | done | [runs/E021-qsa-host-scan.md](runs/E021-qsa-host-scan.md) |
| E022 | 2026-09-17 | T1 | dev-rx9070-16g | can the `O(n_kv)` host scan be avoided | yes: contiguous-run fast path, tg +8.3% @40k / +22.0% @164k | done | [runs/E022-qsa-fast-path.md](runs/E022-qsa-fast-path.md) |
| E023 | 2026-09-17 | T1 | dev-rx9070-16g | does one pass + no divisions help more | only 1-2% more; residual ~1.7 ms is store volume, loop tuning done | done | [runs/E023-qsa-fast-path-one-pass.md](runs/E023-qsa-fast-path-one-pass.md) |
| E024 | 2026-09-17 | T3 | analysis | where the per-QSA-layer GPU work goes | ~76 MB and ~33 nodes per QSA layer per step; not measured in ms | open | [runs/E024-qsa-gpu-inventory.md](runs/E024-qsa-gpu-inventory.md) |
| E025 | 2026-09-17 | T2 | bench-4x-r9700-32g | do the host fix and sparse FA show up on the real model | A yes (+4.9% tg @131k), B null; 20.4 ms/step unexplained by bytes | open | [runs/E025-bench-validation.md](runs/E025-bench-validation.md) |
| E026 | 2026-09-18 | T1 | dev-rx9070-16g | what the user's rebase did to the dummy numbers | gate re-baselined (+0.067%); their mmq retune costs pp512 -11.2% here | done | [runs/E026-post-rebase-rebaseline.md](runs/E026-post-rebase-rebaseline.md) |
| E027 | 2026-09-18 | T2 | bench-4x-r9700-32g | does sparse FA pay on the real model | yes, once PLE faults stop masking it: pp +25.6/+42.3/+45.1% @131k | done | [runs/E027-bench-sparse-engages.md](runs/E027-bench-sparse-engages.md) |
| E031 | 2026-09-18 | T2 | bench-4x-r9700-32g + dev-rx9070-16g | does PR 29030's direct-read gather pay | yes: pp512 @131k +37.2%, the first thing to move bench prefill | done | [runs/E031-lazy-direct-reads.md](runs/E031-lazy-direct-reads.md) |
| E032 | 2026-09-18 | T0 | analysis (reviewer run) | is there an unbounded index in the sparse port | no gate given; fixed the `n_kv_max` latch; 256/256/1/16 had zero ops coverage | done | [runs/E032-audit-sparse-bounds.md](runs/E032-audit-sparse-bounds.md) |
| E033 | 2026-09-18 | T2 | bench-4x-r9700-32g | do the bench GPU faults belong to the sparse port | not attributed: MES timeouts at teardown, no GPUVM faults | not attributed | [runs/E033-bench-gpu-hangs.md](runs/E033-bench-gpu-hangs.md) |
| E034 | 2026-09-18 | T1 | dev-rx9070-16g | does the pulled RDNA3 decode kernel (fattn-rtile) build, engage and pay on gfx1201 | builds and engages but does not pay as dense (-9.5% @40k); gated `GGML_FATTN_RDNA_RTILE` | done | [runs/E034-rtile-decode-kernel.md](runs/E034-rtile-decode-kernel.md) |
| E035 | 2026-09-18 | T1 | dev-rx9070-16g | can decode take the QSA selection at all, and does it pay | yes: sparse decode tg +4.2% @40k / +10.6% @164k on the dummy | done | [runs/E035-sparse-decode.md](runs/E035-sparse-decode.md) |
| E036 | 2026-09-18 | T1 | dev-rx9070-16g | what is decode's context-proportional cost, if not the KV read | the chain itself: 2.78 ms per QSA layer per step; selection runs in the wrong order | done | [runs/E036-qsa-chain-cost.md](runs/E036-qsa-chain-cost.md) |
| E037 | 2026-09-19 | T1 | bench-4x-r9700-32g | which kernels the QSA chain actually is, on the real model | top_k first, 41 launches per layer; 46% removable by H13, 36% by H9 | done | [runs/E037-qsa-chain-kernel-attribution.md](runs/E037-qsa-chain-kernel-attribution.md) |
| E038 | 2026-09-19 | T1 | bench-4x-r9700-32g | does the chain grow linearly with depth | no, it is the run shape; chain device time = 79% of the measured gain | done | [runs/E038-two-depth-decomposition.md](runs/E038-two-depth-decomposition.md) |
| E039 | 2026-09-19 | T1 | dev-rx9070-16g | does block-level QSA selection (H13) pay, and does it cost quality | yes and neutral: +6.3% tg / +6.8% pp @164k | done | [runs/E039-h13-block-selection.md](runs/E039-h13-block-selection.md) |
| E040 | 2026-09-19 | T1 | bench-4x-r9700-32g | what the bench looks like after H13 | H13 alone: tg +4.3% @131k; the pp delta is not H13 alone | done | [runs/E040-bench-after-h13.md](runs/E040-bench-after-h13.md) |
| E041 | 2026-09-20 | T1 | dev-rx9070-16g | background review of H13 plus its fix | two defects fixed (tail seq filter, rank/cell space mix); bits identical | done | [runs/E041-review-of-h13.md](runs/E041-review-of-h13.md) |
| E042 | 2026-09-20 | T1 | bench-4x-r9700-32g | decode-only kernel attribution via rocprofv3 `--selected-regions` | the chain is data movement, not top_k: +10.5 ms/token/GPU vs no-qsa | done | [runs/E042-decode-only-traces.md](runs/E042-decode-only-traces.md) |
| E043 | 2026-09-20 | T1 | reviewer-5 | background review of E042: mechanism correction + fusion survey | the 11.1 ms is re-deriving 32768 static block keys; H9 is output-neutral | done | [runs/E043-review-of-e042.md](runs/E043-review-of-e042.md) |
| E044 | 2026-09-20 | T1 | dev-1x-rx9070-16g | H9 built: pool indexer block keys at write time (`d8bce4e25`, gate `Q4EXP_POOLED`) | bit-identical on every dev gate; two real bugs found only by numeric sweeps | done | [runs/E044-h9-pool-implementation.md](runs/E044-h9-pool-implementation.md) |
| E045 | 2026-09-24 | T1 | bench-4x-r9700-32g | H9 on the real 4-card box + a GPU hang + the rollback hole | tg +30.1% @131k but pp -16..-22%; rollback clamp in `90b9ccf9d` | done | [runs/E045-h9-on-4cards-rollback-hole.md](runs/E045-h9-on-4cards-rollback-hole.md) |
