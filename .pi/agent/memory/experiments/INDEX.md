# INDEX - qwen4exp / RDNA4 experiments

Append-only ledger. One row per experiment. Newest at the bottom (ids never reuse).
Detail lives in `runs/`; this table is for scanning and for answering "did we already
try that?".

| id | date | tier | machine | hypothesis (one line) | headline result | verdict | record |
|---|---|---|---|---|---|---|---|
| E001 | 2026-09-17 | T1 | dev-rx9070-16g | the T1 harness (dummy models + fusion counts + bench) runs on ROCm0 | loader trap was the whole story. `test-backend-ops` passes (1500/1500 non-FA, 3973/3979 FA); 111 dummy models generate; qwen4exp graph runs on GPU incl. `-fa 1`; **`test-fusion` is impossible here - the fusion debug API is Metal-only** | done | [runs/E001-t1-harness-viability.md](runs/E001-t1-harness-viability.md) |
| E002 | 2026-09-17 | T1 | dev-rx9070-16g | a synthetic qwen4exp model gives a usable pp/tg baseline | **no**: 19 MB F32 fits in cache, so pp/tg measure launch + input-path overhead, not the real bottleneck | dead-end | [runs/E002-synthetic-baseline.md](runs/E002-synthetic-baseline.md) |
| E003 | - | T1 | dev-rx9070-16g | qwen4exp pays a context-scaling QSA tax for sparsity it cannot collect on ROCm | - (instrument now blocked on E004; `n_kv_max` flip is inert, see E001) | planned | [runs/E003-qsa-tax.md](runs/E003-qsa-tax.md) |
| E004 | - | T1 | dev-rx9070-16g | a config-shaped synthetic model (real head_dim 256 / 24-2 heads / hc_lowrank 320 / budget 2048, fewer experts) is a usable pp instrument | - | planned | (record not yet written) |
| E005 | 2026-09-17 | **T2** | bench-4x-r9700-32g | real qwen4exp Q4_K_M on 4x R9700 gives us a baseline | pp512/4k/8k = 397/512/547 t/s, tg128 = 28.2 t/s; **~9% GPU util**; **not a baseline** - user's fork, different ROCm | done | [runs/E005-real-baseline.md](runs/E005-real-baseline.md) |
| E006 | 2026-09-17 | T2 | bench-4x-r9700-32g | `-sm tensor` costs tg via per-layer cross-GPU collectives (no Infinity Fabric on Navi 48) | **refuted by sign**: layer split does ~zero collectives and is *slower* (tg 24.7 vs 28.2); also refutes bandwidth-bound (predicted 0.25 ratio, measured 0.88). tg is serial and mode-independent | done | [runs/E006-split-mode.md](runs/E006-split-mode.md) |
| E007 | - | T2 | bench-4x-r9700-32g | move the n-gram table onto VRAM to avoid the host gather | **impossible**: the PLE path is *mirrored* under `-sm tensor` (`src/llama-model.cpp:513-515`), so ~30 GB becomes ~30 GB/card on a 128 GB box | dead-end | folded into [plans/ple-prefetch.md](plans/ple-prefetch.md) |
| E008 | 2026-09-17 | T2 | bench-4x-r9700-32g | the CPU-placed table might defeat HIP graph capture | **no** - capture is active: graphs off costs 7% of tg and ~7% of deep pp. Bounds total launch-submission cost at ~2.7 ms/token | done | [runs/E008-graph-capture.md](runs/E008-graph-capture.md) |
| E009 | 2026-09-17 | T2 | bench-4x-r9700-32g | the graph may be reallocated at unchanged size every step | **no** - the hook GGML_ABORTs when it fires and the run finished clean; same-size realloc ruled out | done | [runs/E009-graph-realloc.md](runs/E009-graph-realloc.md) |
| E010 | 2026-09-17 | T2 | bench-4x-r9700-32g | graph reuse may already be broken for this model | **no, best number yet**: reuse works, worth ~22 ms/step (tg 28.20->17.36; pp512 +19.6 ms on its single build); pp insensitive, decode collapses | done | [runs/E010-graph-reuse.md](runs/E010-graph-reuse.md) |
| E011 | 2026-09-17 | T2 | bench-4x-r9700-32g | if tg scales super-linearly with concurrency the cost is per-step host work | **yes**: ~28 ms/step is fixed regardless of tokens (B=1/2/4 -> 32.4/36.7/52.5 ms per step); ~90% of a decode token is per-step cost. Bonus: 40k depth costs only ~10% of tg, bounding H4b | done | [runs/E011-concurrency.md](runs/E011-concurrency.md) |
| E014 | - | T2 | bench-4x-r9700-32g | `-lzm off` + `-ot ...=CPU` gives a resident table with no demand paging; if tg improves, faults are part of the fixed ~28 ms | - | planned | [runs/E011-concurrency.md](runs/E011-concurrency.md#found-in-the-header-of-this-log--ot-has-never-done-anything) |
| E015 | 2026-09-17 | T2 | bench-4x-r9700-32g | scheduler op-offload may be the fixed per-step cost | **no** - tg 28.14 vs 28.20, pp 547.5 vs 546.8, CSV confirms `no_op_offload=1` applied | dead-end | [runs/E015-no-op-offload.md](runs/E015-no-op-offload.md) |
| E017 | 2026-09-17 | T1 | dev-rx9070-16g | a shape-faithful dummy can be generated and run locally | **yes**: 4 real layers + real types, 5.47 ms/step of which ~4.5 ms is per-step host work; decode is flat across a 10x table volume range (182.1/182.7/182.4) while pp4096 gains 6.5% from a small table; shipped as models/q4exp-4l.gguf with a 5.5 GB table | done | [runs/E017-dummy-harness.md](runs/E017-dummy-harness.md) |
| E018 | 2026-09-17 | T1 | dev-rx9070-16g | a zero-filled dummy cannot detect a model-level bug | **yes**: seeded pattern fill + `llama-perplexity` fingerprint, `PPL = 262938.7619 +/- 3039.07`, bit-stable across runs, and perf-neutral vs zeros (181.89 vs 182.40, -0.28%); local noise floor measured at ~1% so only >2% deltas count | done | [runs/E018-correctness-gate.md](runs/E018-correctness-gate.md) |
| E019 | 2026-09-17 | T1 | dev-rx9070-16g | the op-level gate needs a baseline on the current stack | **yes**: 5641 cases, 5633 OK / 7 FAIL, all FAILs pre-existing `FLASH_ATTN_EXT hsk=192,hsv=128`; our 256/256 shape is 142/142. Capture works locally now (9430 warmups); `-j 8` aborts, use `-j 1` | done | [runs/E019-ops-baseline.md](runs/E019-ops-baseline.md) |
| E020 | 2026-09-17 | T1 | dev-rx9070-16g | can sparse FA be made to run on RDNA4 | **yes**: pp512 +23.3% @40k and +58.5% @164k, tg flat, numerics match dense to 6e-7. Six blockers, all mapped; RDNA needs the 1x16 tiling (no device code below 16 tiles) and the generator pruned it. Also closes H4b's kernel question | done | [runs/E020-sparse-fa-rdna4.md](runs/E020-sparse-fa-rdna4.md) |
| E021 | 2026-09-17 | T1 | dev-rx9070-16g | what actually owns the decode depth slope | **the host-side QSA input scan**: O(n_kv) at 23 ns/cell, 695 us/step at 30k = 13% of the step and ~2/3 of the depth delta. Matches upstream's own TODO. Indexer block keys are not cached (pooling precedes norm/rope), and the mapping is append-only, so it is fixable without kernel work and does not scale with layer count | done | [runs/E021-qsa-host-scan.md](runs/E021-qsa-host-scan.md) |
| E022 | 2026-09-17 | T1 | dev-rx9070-16g | can the O(n_kv) host scan be avoided | **yes**: contiguous-run fast path, tg +8.3% @40k and +22.0% @164k, pp unchanged, all four correctness arms bit-identical. First local change that moves decode. Also found: the `-np 2` gate arm is not reproducible (~1e-7) | done | [runs/E022-qsa-fast-path.md](runs/E022-qsa-fast-path.md) |
| E023 | 2026-09-17 | T1 | dev-rx9070-16g | does one pass + no divisions help more | yes but little: +1.3%/+2.1% then +0.1%/+1.2%; cumulative vs pre-E022 **tg +9.8% @40k, +26.0% @164k**, gates still bit-identical. Residual ~1.7 ms/step is the **store volume** (~3 MB of mapping at 164k, ~3 ns/store), so loop tuning is done; only reusing the arrays instead of rewriting them helps | done | [runs/E023-qsa-fast-path-one-pass.md](runs/E023-qsa-fast-path-one-pass.md) |
| E024 | 2026-09-17 | T3 | analysis | where the per-QSA-layer GPU work goes | ~76 MB and ~33 graph nodes per QSA layer per step, of which ~50 MB and ~9 nodes is re-pooling the whole indexer cache; that is 2x dense attention and ~19x sparse attention after E020, and it is the half that scales with layer count (12x real). Not measured | open | [runs/E024-qsa-gpu-inventory.md](runs/E024-qsa-gpu-inventory.md) |
| E025 | 2026-09-17 | T2 | bench-4x-r9700-32g | do the host fix and sparse FA show up on the real model | **A yes**: +4.9% tg at 131k, 2.34 ms/token saved, 20.1 ns/token of depth = 1x not 4x, so the fill is per step and per rank hypothesis is dead. **B null everywhere** - it did not engage (locally it was +23%/+58% pp). And 20.4 ms of a 50 ms step is depth-proportional while bytes explain ~1/9 of it | open | [runs/E025-bench-validation.md](runs/E025-bench-validation.md) |
| E026 | 2026-09-18 | T1 | dev-rx9070-16g | what the user's rebase did to the dummy numbers | fingerprint moved +0.067% (their mmq retune, legitimate: dense and sparse both 0.046%), gate re-baselined; qsa-A (+9.8%/+26%) and qsa-B (+21%/+55% pp) intact, dense-vs-sparse still 5e-8. Their retune costs pp512 -11.2% here | done | [runs/E026-post-rebase-rebaseline.md](runs/E026-post-rebase-rebaseline.md) |
| E027 | 2026-09-18 | T2 | bench-4x-r9700-32g | does sparse FA pay on the real model | **yes, once the stall in front of it is fixed** (third correction): same build 11071, `-lzm on-direct`, dense vs sparse -> +12.4/+15.4/+17.8% at d40960 and **+25.6/+42.3/+45.1%** at d131072, tg flat. The 11062 control's +-0.34% was PLE demand faults masking it; combined at pp4096@131k it is 759 -> 847 -> 1205, +59% | done | [runs/E027-bench-sparse-engages.md](runs/E027-bench-sparse-engages.md) |
| E031 | 2026-09-18 | T2 | bench-4x-r9700-32g + dev-rx9070-16g | does PR 29030's direct-read gather pay | **yes, and it is the first thing that moves bench prefill**: pp512 @131k 476.2 -> **653.2 (+37.2%)**, pp4096 +9.4%, pp8192 and tg flat. Dev box +22.3% pp512 with golden PPL bit-identical. H11 confirmed | done | [runs/E031-lazy-direct-reads.md](runs/E031-lazy-direct-reads.md) |
| E032 | 2026-09-18 | T0 | analysis (reviewer run) | is there an unbounded index in the sparse port | **not given the gate**, which moves the suspicion to the unclamped gather index, the zero-slack LDS in the new 256/256/1/16 instantiation, and an unenforced 16B alignment precondition. Found instead: my `static n_kv_max` latch (fixed, `1d724aca0`), and that this instantiation has **zero** coverage in test-backend-ops, so E019 never validated it | done | [runs/E032-audit-sparse-bounds.md](runs/E032-audit-sparse-bounds.md) | (local repro at 6 QSA layers with the real head geometry does not fault; content or multi-device left)
| E033 | 2026-09-18 | T2 | bench-4x-r9700-32g | do the bench GPU faults belong to the sparse port | **no support for that any more**: dmesg shows MES timeouts at queue teardown on one device with zero GPUVM faults, i.e. hangs not illegal accesses, and the last incident was at `n_ctx=4608` where sparse cannot engage. It followed the `HIP_LAUNCH_BLOCKING` deadlock I recommended; after a power cycle the full matrix ran clean twice, dense and sparse | not attributed | [runs/E033-bench-gpu-hangs.md](runs/E033-bench-gpu-hangs.md) |
| E034 | 2026-09-18 | T1 | dev-rx9070-16g | does the pulled RDNA3 decode kernel (fattn-rtile) build, engage and pay on gfx1201 | **builds and engages, does not pay as a dense kernel**: fixed the inverted .cu/.cuh pair, the 12-vs-11 arg `launch_fattn` call (added a defaulted `min_parallel_blocks`), the missing hook cases in `fattn.cu`, RDNA4 acceptance, and gated it on `GGML_FATTN_RDNA_RTILE`. FA suite has the same 6 known failures on and off; golden PPL bit-identical; tg 200.07 vs 200.27 at 40k and 126.14 vs 126.16 at 164k. Its KV-split floor *hurts* here (-9.5% at 40k with PB=300, -20% at 164k with PB=144), and PB moving the number is the engagement proof | done | [runs/E034-rtile-decode-kernel.md](runs/E034-rtile-decode-kernel.md) |
| E035 | 2026-09-18 | T1 | dev-rx9070-16g | can decode take the QSA selection at all, and does it pay | **yes now**: taught fattn-rtile to walk the index list (`use_sparse` as a template parameter, -1 and short-tail padding), and tg on the dummy with both arms sparse is 199.20 -> 207.63 at 40k (**+4.2%**) and 125.99 -> 139.40 at 164k (**+10.6%**), implying ~420 GB/s on the removed reads. 6 rtile-eligible sparse decode cases in test-backend-ops pass, which is the coverage E032 said never existed. Predicts +20-30% tg at 131k on the bench | done | [runs/E035-sparse-decode.md](runs/E035-sparse-decode.md) |
| E036 | 2026-09-18 | T1 | dev-rx9070-16g | what is decode's context-proportional cost, if not the KV read | **the QSA chain itself**: dropping `build_qsa_top_k` (dense attention, wrong output, cost only) takes tg 125.80 -> 193.36 at 164k, i.e. the chain costs **2.78 ms per QSA layer per step** while sparse decode saves 0.83 ms. Ratio 3.3:1, grows with depth, and it explains the bench null. Also found the selection runs in the wrong order vs the paper - top-k over expanded cells instead of over blocks, 4x the data through the biggest terms | done | [runs/E036-qsa-chain-cost.md](runs/E036-qsa-chain-cost.md) |
| E037 | 2026-09-19 | T1 | bench-4x-r9700-32g | which kernels the QSA chain actually is, on the real model | **top_k first, and 41 launches per layer**: chain on vs off (only `Q4EXP_NO_INDEXER` differs, rtile off so FA work is identical) = +14.775 s device time and +844,740 launches; ranked top_k 3.12 s / get_rows 2.91 / adds 1.38 / rope 1.34 / cont-permute 1.18 / strided slices 0.91. Grouped: 46% removable by H13, 36% by H9, 20% by neither. Retracts E036's top_k demotion. Proves the indexer cache is f16 via `llama-perplexity -v` | done | [runs/E037-qsa-chain-kernel-attribution.md](runs/E037-qsa-chain-kernel-attribution.md) |
| E038 | 2026-09-19 | T1 | bench-4x-r9700-32g | does the chain grow linearly with depth | **no, and the reason is the run shape**: prefill builds 160->512 (=4 passes x depth/1024) and decode builds 1540 unchanged (3 chain builds per token), so the x6.5 growth is 71 s prefill vs 25 s decode at 131k. Chain device time = 12.2 ms/token = 79% of the measured 15.5 ms gain, closing the dispatch hypothesis. Per-card cost says the chain is split ~4 ways -> H17 (per-device selection may not be the reference's top-k) | done | [runs/E038-two-depth-decomposition.md](runs/E038-two-depth-decomposition.md) |
| E039 | 2026-09-19 | T1 | dev-rx9070-16g | does block-level QSA selection (H13) pay, and does it cost quality | **yes, neutral on quality**: +6.3% tg / +6.8% pp at 164k, +2.3% / +2.0% at 40960, flat at 8192, same-binary A/B via `Q4EXP_CELL_SEL`; deep-arm PPL moves 8e-8 and the shallow golden corpus stays bit-identical because 512 cells is under the 2048 budget, where both paths must agree. Deletes the `n_kv` expand, its two copies, the per-cell mask add, top-k over `n_kv`, and the `cell_blk` forward map with its host fill. Also records the first A/B I ran, which read +86% / +46% and was bogus because the gate compared against the *fallback* path - caught by the baseline disagreeing with E036 | done | [runs/E039-h13-block-selection.md](runs/E039-h13-block-selection.md) |
| E040 | 2026-09-19 | T1 | bench-4x-r9700-32g | what the bench looks like after H13 | **H13 alone on the bench = tg +4.3% at 131k (1.8 sigma), flat inside noise at 4096/16384/40960** - `git log d77fb8d53..b827606c8` is exactly one commit, so that comparison is clean. The +22.5% pp in the same log is NOT H13 alone: arm B's build predates the `n_kv_max` latching fix too, and rtile cannot move prefill, so prefill is H13 plus `1d724aca0` jointly. Suggests the sparse prefill path was crippled by that bug rather than by the kernel work. One run with `Q4EXP_CELL_SEL=1` splits it | done | [runs/E040-bench-after-h13.md](runs/E040-bench-after-h13.md) |
| E041 | 2026-09-20 | T1 | dev-rx9070-16g | background review of H13 plus its fix | **two defects fixed** (D1: tail had no sequence filter, per-seq key -> cell map; D2: rank/cell space mix, fallback deleted), exercised paths bit-identical, and the user's post-review bench matches the 11098 build within 0.8% everywhere (pp512 aside, ~9% noise and larger than the deltas). Evidence corrected: same-binary cell arm 267035.3047 replaces the pre-commit 267035.3653, PPL gate blind to selection (~3e-10 not 8e-8), gate 4102/4104 with no engagement change at any depth, `!blk_bias` fallback unreachable. The post-review bench adds nothing on the E040 decode sign (no cell arm). Owed: selection differential, cell-gate A/B, gate split | done | [runs/E041-review-of-h13.md](runs/E041-review-of-h13.md) |
| E042 | 2026-09-20 | T1 | bench-4x-r9700-32g | decode-only kernel attribution via rocprofv3 --selected-regions | **chain is data movement, not top_k**: per-token-per-GPU, qsa adds +10.5 ms vs no-qsa (corrected from 8.7 by E043); top_k trio = 0.55 (H13 did its job); the rest is the indexer re-deriving the static pooled block keys (gather 3.0 + rect copies 3.0 + rope 2.5 + norm 1.1 + adds 1.5). Both arms pay NCCL 4.6/5.9 ms. Needed llama-bench roctx patch + --marker-trace | done | [runs/E042-decode-only-traces.md](runs/E042-decode-only-traces.md) |
| E043 | 2026-09-20 | T1 | reviewer-5 | background review of E042: mechanism correction + fusion survey | **the 84%-redundant core**: nothing materializes a 2048-cell slice (E042's mechanism wrong); the 11.1 ms is the indexer re-deriving all 32768 pooled block keys per token, all inputs static once a block completes, so H9 (cache at write time) is output-neutral with a measured 35% ceiling. Fusion matcher can't reach the graph (mrope gate + node adjacency); bespoke kernel is step 3, after H9 and two cheap launch-config fixes (rope_multi 64/256 threads, mask compaction single workgroup) worth ~2 ms. F8 unresolved: FA rows at 6/token/GPU, half the expected 12 | done | [runs/E043-review-of-e042.md](runs/E043-review-of-e042.md) |

## Comparability breaks

Any row here means numbers on either side of it are not valid A/B partners.

| date | machine | change | invalidated |
|---|---|---|---|
| 2026-09-17 | both | **the two boxes are not the same stack**: bench is ROCm 7.15.0 / Fedora 44 / kernel 7.2.5, dev **was** ROCm 6.4.4 / Fedora 43. Dev was upgraded to **ROCm 7.1.1 / Fedora 44 / kernel 7.2.5-200** at 17:47 the same day (hw profile v2), so the ROCm gap narrowed but GPU count and code state still differ | no dev-box number may be an A/B partner for a bench-box number, and vice versa. E005's numbers additionally come from a **fork** (`c9a59ef73`) with RDNA4 MMQ fixes, a custom AllReduce, and qwen4exp tensor-split enablement that this branch does not have |

## Machine profiles

| profile | role | state |
|---|---|---|
| [hw/dev-rx9070-16g.md](hw/dev-rx9070-16g.md) | T1: build, op-level, synthetic models | **v2** since 2026-09-17: ROCm 7.1.1, Fedora 44. **All v1 dev numbers predate the stack change and its `build/` is unloadable** - reconfigure before any T1 work |
| [hw/bench-4x-r9700-32g.md](hw/bench-4x-r9700-32g.md) | T2: real weights, end-to-end, multi-GPU | v1: gfx1201 confirmed, 4x32 GB = 128 GB. Runs Q4_K_M on GPU + Q5 PLE table in RAM. ROCm ver / topology / sys RAM still unknown |

## Open threads (not yet experiments)

Tracked in [plans/backlog.md](plans/backlog.md). Promote a thread to an `E<nnn>` id
only when it has a falsifiable hypothesis written down.

Substrate reference: [plans/model-shape.md](plans/model-shape.md) - real `qwen4exp`
dimensions vs what the synthetic dummy uses, and the list of places where the dummy's
value sits on a kernel-selection boundary. Needs the HF `config.json` (text only, no
weights) to fill. **Filled 2026-09-17.**

Design reference: [plans/ple-prefetch.md](plans/ple-prefetch.md) - the n-gram table
prefetch/cache options (warm page cache; VRAM row cache with an S3-FIFO/SIEVE-class policy;
why the whole table cannot go on GPU), and what has to be measured before any of it is built.
**Superseded in premise by E031**: direct per-row reads removed the fault path, so the row-cache
options are now solutions to a problem that no longer exists. Read the newer file first.

Architecture walkthrough: [plans/qsa-ple-in-prefill-and-decode.md](plans/qsa-ple-in-prefill-and-decode.md)
- what QSA and the n-gram table actually execute in prefill versus decode, per-token byte budgets at
this model's real shapes, why the sparse port is structurally prefill-only here, and what a smarter
table cache could and could not buy. Written to decide priorities, not to record a run.

Implementation plan: [plans/h13-block-selection.md](plans/h13-block-selection.md) - block-level QSA
selection as the paper does it, with the host-side `tail_cells` list that the spare-bucket trap forces,
why the change is not equality-testable, and the six validation steps. Blocked on a go/no-go from the
user, since it changes which cells attention sees.

## Findings that are not experiments
- **E019's ops baseline says nothing about the sparse path (E032):** the `256/256` sparse cases dispatch to
  `ncols2=8` on NVIDIA or to VEC at `nb=1`, so the `(256,256,1,16)` instantiation the RDNA4 port had
  to add is unreachable on every NVIDIA path in the tree and untested everywhere. Gate-probe output
  is the only proof of engagement, and 12 < 16 (`gqa_ratio < ncols2`) has never run anywhere else.


Things established while setting up, recorded here because they change what is worth
measuring at all. Detail in the topic memories.

- **Sparse FA does not exist on ROCm, and qwen4exp still pays for it.** The
  indexer plus a per-layer mask rebuild run on every full-attention layer, but
  `build_attn_mha` passes `n_kv_max = 0`, so the kernel iterates the whole KV extent.
  Switching it on is not possible on this backend today:
  `ggml_cuda_flash_attn_ext_compact_mask` does `GGML_ABORT("sparse flash attention is only
  supported on NVIDIA CUDA")` under `GGML_USE_HIP`
  (`ggml/src/ggml-cuda/fattn.cu:93-97`). Drives E003, H4a, H4b.
- **`qwen4exp` is a 180 B MoE, and 28% of it is one hash-gathered embedding table.**
  51.2 B params of PLE n-gram table serving a single layer, run as Q5 in host RAM by the
  user; `-lzm auto` lazy-reads any tensor over 4 GiB
  (`src/llama-model-loader.cpp:1093`), so "in RAM" may mean "page-faulted on demand".
  Backlog L1/L2. Full shape in [plans/model-shape.md](plans/model-shape.md).
- **`test-fusion` does not exist for this backend.** The generic fusion debugging API
  (`ggml_backend_fusion_*`) is implemented only in `ggml/src/ggml-metal/ggml-metal.cpp` `[v]`,
  so `--device ROCm0` refuses to run and `tests/fusion/` has exactly one baseline CSV. This
  kills P3 and removes the natural instrument for N1 (the IMROPE fusion gap). Backlog T1.
- **A synthetic qwen4exp model is not a perf baseline** (E002, dead-end): 19 MB F32 fits in
  cache, so pp/tg report harness overhead. It *is* a valid structural instrument - the graph
  loads and runs on `ROCm0`, with and without `-fa 1`.
- **Dummy GGUFs abort anything that tokenizes** - `llama-vocab.cpp:3393` assert, no
  tokenizer in the file. Use token-id consumers (`llama-bench`, `test-save-load-state`).
- **`-sm tensor` throws on this branch for qwen4exp but is what the bench box actually
  runs**: `llm_arch_supports_sm_tensor` returns false (`src/llama-arch.cpp:1161`) and
  `llama_model_create` raises (`src/llama-model.cpp:358`); the user's fork enables it, and
  E006 measured tensor split winning decode. So this is an obstacle for B4, not a hardware
  limitation - and I previously had it backwards, claiming the box was stuck on layer split.
- The dev box's `~/.local/lib64` shadows the working tree for every dynamic binary, and it
  already produced two false failures. Drives B0.

- **`test-backend-ops` is not broken on AMD.** On gfx1201 / ROCm 6.4.4 (dev hw v1) it passed 1500/1500
  non-FA and 3973/3979 FA cases with no hang (E001 update). The 6 failures are all
  `hsk=192/hsv=128`, a shape qwen4exp does not use. So B1 is closed for the dev box and the
  bench-box hang is a box-specific symptom needing a different explanation - 4-GPU config,
  ROCm version, or code state, all B2 unknowns.
- **Flipping `n_kv_max` on ROCm is inert, not risky.** Sparse-shaped FA cases report
  `SUPPORTED` and then silently compute dense, because `use_sparse` is false on HIP and
  `compact_mask` is never reached. H4b is kernel work; it cannot be staged from the model
  side. See E001 stage 3.
- **Sparse FA support at qwen4exp's shape is empirically fine on ROCm0:** 142/142
  `FLASH_ATTN_EXT` cases at hsk/hsv 256/256 supported (E001 support probe).

- **The box is nowhere near a hardware floor, so overhead is the project.** E005: pp8192
  reaches ~3.3 TFLOP/s against a user-reported ~180-190 TFLOPS WMMA fp16 peak, and `GFX` util
  sampled at **9% on all four cards**. tg128 = 35.5 ms/token vs a ~1-2 ms bandwidth floor.
  Consequence: per-kernel work (H4b/QSA) is demoted; cross-GPU sync, host-side stalls and
  placement (E006, F1-F3) are promoted.
- **`size_label` is a lie, structurally**: it is parsed from the repo/file name
  (`gguf-py/gguf/metadata.py:314-328`), so this GGUF reported `A3B` for a ~6 B-active model.
  Derive params yourself (backlog F2).
- **`-lm none` does not disable lazy reads, and no load mode ever prefetches the lazy table**
  - the WILLNEED loop excludes lazy ranges and they are marked `MADV_RANDOM`
  (`src/llama-mmap.cpp:500-510`). Backlog F1.

- **Four mechanisms for the tg gap have now been eliminated by measurement, not argument**:
  weight bandwidth (E006 ratio 0.88 vs 0.25 predicted), cross-GPU collectives (E006, refuted
  by sign), split-mode choice (E006), and graph capture / launch submission (E008, worth only
  ~2.7 ms of a 35.5 ms token). PLE page faults are bounded at ~0.1-4 ms by arithmetic. What is
  left is per-step **host-side graph build/alloc**, which graph replay cannot remove - hence
  E009/E010/E011.

- **`-ot per_layer_token_embd=CPU` has never applied** - lazy-read tensors ignore tensor
  overrides (`W llama_model_loader: tensor overrides do not apply to lazy-read tensors`), and
  lazy mode itself forces the CPU buffer type (`src/llama-model-loader.cpp:1080-1086`). The
  placement is real; the mechanism was not what F3 claimed. See E011.

- **A host-side graph pass costs ~20-22 ms on this box** (E010): disabling graph reuse
  dropped tg 28.20 -> 17.36 while every pp row stayed within noise, and pp512's whole test
  gained 19.6 ms for its single build. Two independent measurements bounding the same number
  is why it is trusted. Per-step cost is invisible in prefill (amortised over ~512 tokens) and
  dominant in decode - so **pp numbers must never be used to reason about decode**.
- **Six mechanisms for the ~28 ms/step are now excluded**: weight bandwidth, cross-GPU
  collectives, split mode, graph capture, scheduler op-offload (E015), same-size graph realloc
  (E009). Reuse works. What remains is per-step host work that flags cannot reach -> E016 is a
  host profile, not another A/B.

- **The qwen4exp n-gram table is 160 x 320,001,536 stored as Q5_0** (real file dump, E017).
  Earlier notes described it as `20M rows x ple_embed_dim 2560`; the total (51.2 B params) was
  right but the structure was not: `ple_embed_dim` is the **concatenated** per-token width
  (16 heads x 160), so a row is 160 elements and llama.cpp requires
  `embedding_length_per_layer_input * ple_n_heads == n_embd`.
- **The bartowski Q4_K_M file is mixed precision, not uniform Q4** (E017): attn_k/v are Q8_0,
  attn_output Q5_K, the router and all HC/PLE gammas are F32 (stored flat, 10240 elements),
  hc_*_inject and the indexer projections are BF16, hc_*_up and ffn_down_exps Q5_0, and
  attn_q / ssm_out / shexp vary per layer. llama-quantize's own tables would not reproduce
  it, so the harness writes these types directly.

- **`models/q4exp-4l.gguf` is seeded-filled, not zero-filled, and `llama-perplexity` on
  `tools/golden-corpus.md` is the model-level correctness gate** (E018). The diff contract is
  'every diff explained', not 'no diff': the QSA compaction port must move the logits.
- **Local deltas below ~2% on the dummy harness are noise** (E018: same file measured
  180.3/181.8/181.9 t/s across invocations). Iterate on 1.3 GB smoke builds; the 12.76 GB
  build takes ~40 s of I/O and should be written once.

- **The op-level baseline on ROCm 7.1.1 is 5633 OK / 7 FAIL** (E019), the FAILs all being the
  known `FLASH_ATTN_EXT hsk=192,hsv=128` family. New failures are judged against 7, not the
  6 in the older note. `test-backend-ops -j >1` aborts on HIP capture errors - run `-j 1`.

- **Sparse FA works on RDNA4 (E020)**: `Q4EXP_SPARSE_FA=1` engages above 4102 KV, gives
  +23% pp at 40k / +58% at 164k, and matches dense numerics to 6e-7. Decode does not move.
  The E018 golden corpus is too short to reach the depth gate - use `tools/sparse-corpus.md`
  with `-c 8192` for any sparse-path check.

- **The decode depth slope is mostly host-side** (E021): `set_input_qsa` scans all n_kv cells
  per step (~23 ns/cell, 13% of a 30k step). This is per-step, NOT per-layer, so unlike every
  other local number it is not 1/12 attenuated by the dummy harness - a fix here is directly
  meaningful on this box and the same absolute cost on the bench.

- **Decode moved for the first time (E022)**: `set_input_qsa` now has a verified contiguous-run
  fast path, tg +8.3% at 40k / +22.0% at 164k, pp unchanged, results bit-identical. It is
  host-side and per-step, so the bench box should see the same absolute saving.
  Remaining: one-pass version (~2x again), memoization (near O(1)), and the ~1/3 of the slope
  that is GPU-side O(n_kv) mask/top-k work.
- **`llama-perplexity -np > 1` is not bit-reproducible** (~1e-7 spread, shared cache ordering).
  Use `-np 1` arms for exact-match claims (E022).

- **QSA host scan cumulative (E022+E023): tg +9.8% at 40k, +26.0% at 164k**, results bit-identical.
  The remaining ~1.7 ms/step in that function is the cost of *writing* ~3 MB of mapping data
  each step - loop shape and divisions are no longer the limit (E023).
- **That fill is once per step, NOT once per QSA layer** (inputs are shared by compression ratio),
  so the saving is an absolute ~0.8-2.1 ms: ~2-6% of the bench's 35 ms step, not +26%. The QSA
  *GPU* work is per layer and therefore 12x on the real model - the bench-relevant target, and the
  opposite direction from every other local number here. Neither is measurable on a 4-layer dummy.

- **Bench-confirmed (E025): qsa-A is +4.9% tg at 131k and ~0-1% below 40k**, an absolute
  ~20 ns/token saving that is 1x per step (not per rank, not per layer). No pp effect.
- **qsa-B (sparse FA) is a null on the bench and needs an engagement probe** before it can be
  interpreted - locally it gave +23%/+58% pp, so 'not running' is the working hypothesis.
- **~20.4 ms of a 50 ms decode step at 131k is depth-proportional and bytes explain ~1/9 of it**
  (E025) - the QSA GPU path is latency/inefficiency-bound, which is the real target now.

- **Gate values re-baselined at `a8b24dfdf` (E026)**: golden `263113.6984 +/- 3043.13362`; deep
  corpus `-c 8192` dense `267035.3653` / sparse `267035.3524`. Older numbers belong to builds
  before the user's mmq retune, which moved dequant numerics by ~0.05%.
- **Their RDNA4 mmq retune costs ~11% of pp512 on the dummy** (E026, both arms; -4.8% at 164k)
  with decode unchanged. Check against the bench rather than dismiss - but the dummy weights
  pp toward MoE matmuls, which is exactly what was retuned.

- **Sparse FA pays on the bench once the stall in front of it is gone (E027 correction 3):** on
  build 11071 with `-lzm on-direct`, dense vs sparse gives +12 to +18% at d40960 and +26 to +45% at
  d131072, tg flat. The 11062 control's +-0.34% was PLE demand faults masking the attention savings,
  which is a cautionary tale about controls run behind a bottleneck: a null can mean the change is
  worth nothing, or that the thing you changed is not the thing that was limiting you.
- **The bench's fixed per-step cost has moved** (E027 base drift: +4.5% tg at 4k depth). E011-era
  percentages should be re-read against a fresh baseline before being quoted again.

- **Open contradiction (H11): prefill at 20-30% GPU util, decode near 100%, on the bench, both pre-**
  **and post-rebase** (user report) versus E005's 9% during decode on the same box. Utilisation
  counts resident kernels, so a spin-waiting AllReduce reads as 100% busy. Needs the sampling
  method pinned down before it can be used as evidence of anything.

- **H11 screenshots confirm waiting, not work**: four cards at 26% util, ~70 W each, load average
  ~1.2, VRAM 22/32 GiB per card. Disk is busy at ~66 MB/s of **reads** - which at 4 KiB pages is
  ~16k accesses/s, i.e. IOPS/latency-bound random single-page reads from the PLE table, not a
  bandwidth limit. That is what `MADV_RANDOM` does to a scattered gather, and it is why F1
  (predictable row set per ubatch -> `WILLNEED`) is back as the fix worth trying. Swap involvement
  is unproven: the 7 GiB figure is the swap size.

  **Size from the dump: the table is 51.2e9 elements = 35.2 GB = 32.8 GiB in Q5_0**, 29% of the file,
  so the story is eviction pressure rather than "too big for RAM" - it would fit in the 62.7 GiB
  alone, but the file is 111 GiB and this is the largest thing competing for the cache.
  Table ranges get `MADV_RANDOM` with no `MAP_POPULATE` and no `WILLNEED` (F1), so prefill faults
  per access. E017 showed the shape on the dummy (+6.5% pp, 0% tg from a small table); E008b
  (majflt/s in both phases) settles it, E030 checks the llama-bench amplification, and **H12 is
  the structural fix: split across the four cards at ~12 GiB each against ~10 GiB free, which
  removes the streaming instead of batching it.**

- **H11 is confirmed and largely fixed by upstream PR 29030 (E031):** `--lazy-mode on-direct`
  gathers an ubatch's PLE rows with sorted parallel `pread`s instead of demand-faulting the mmap one
  page per ~110 B row. Bench pp512 @131k goes 476.2 -> 653.2 (+37.2%), pp4096 +9.4%, pp8192 and tg
  flat, so prefill was fault-bound and long prompts are now past the crossover into compute-bound.
  The negative on tg is informative too: a decode step only touches 16 rows per PLE layer, so the
  E011 fixed host cost is something else and stays in the queue.
- **`-lm none` does not disable lazy reads (E031):** the lazy ranges are mapped for the table anyway,
  which is how a 111 GiB model loads on a 62.7 GiB box. Do not treat `lm` and `lzm` as one axis.

## Status legend

`planned` (id reserved, nothing run) | `running` | `done` | `blocked` (external
condition missing) | `dead-end` (hypothesis rejected, kept so nobody retests blind)
