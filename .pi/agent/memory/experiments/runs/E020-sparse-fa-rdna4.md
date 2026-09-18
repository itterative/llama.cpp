# E020 - sparse flash attention runs on RDNA4, and it is a prefill win

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- code: `ggml/src/ggml-cuda/fattn.cu`, `fattn-mma-f16.cuh`, `template-instances/`, `src/models/qwen4exp.cpp`
- corpus for the numeric check: `tools/sparse-corpus.md` (120000 B, pinned copy)

## result

Sparse FA (the `n_kv_max` / mask-compaction path) now executes on gfx1201 and computes the same
thing as dense-with-mask. Same binary, only the env differs:

| depth | pp512 dense | pp512 sparse | tg128 dense | tg128 sparse |
|---|---|---|---|---|
| 40960  | 6054 +/- 95   | **7467 +/- 142 (+23.3%)** | 181.80 +/- 0.78 | 181.86 +/- 0.92 |
| 163840 | 2532.6 +/- 3.8 | **4013.3 +/- 46.7 (+58.5%)** | 99.34 +/- 0.01 | 99.59 +/- 0.13 |

Correctness: `llama-perplexity -c 8192` on `tools/sparse-corpus.md` gives **267157.4202** dense vs
**267157.2589** sparse, a 6e-7 relative difference - the same math, different reduction order.

**Prefill gains scale with depth, decode does not move at all.** That is exactly what E011
predicts: sparse FA deletes KV read traffic (from `n_kv` rows down to `n_kv_max` per query row),
and locally decode is fixed-cost bound with only 1 of 4 layers being full attention - about
42 MB/token, roughly 0.08 ms of a 5.5 ms step. So H4b is a long-context prefill optimization here,
not the decode fix.

## the chain of blockers (each one cost a build)

1. **Model side**: qwen4exp builds the top-k mask but passes `n_kv_max = 0`. The sparse call is
   already written and commented out upstream with `TODO: enable sparse attention when we are
   ready` (`src/models/qwen4exp.cpp:761-767`). Enabled behind `Q4EXP_SPARSE_FA`, default off.
   Note the value is **2051, not the configured top_k 2048** - the driver adds a few local slots.
2. **`shall_use_sparse` returned false on HIP** and the `compact_mask` wrapper `GGML_ABORT`ed with
   "sparse flash attention is only supported on NVIDIA CUDA". Both kept for MUSA only now.
3. **The compaction kernel was excluded by `#if !defined(GGML_USE_HIP)`.** PDL is *not* why:
   `ggml_cuda_pdl_sync()` / `pdl_lc()` already compile to nothing on HIP, so the data dependency is
   carried by stream ordering alone. The real compile error was `__ballot_sync`, which on HIP
   requires a 64-bit mask argument and returns 64 bits (`static_assert sizeof(mask) == 8`); added a
   `ggml_cuda_ballot()` wrapper, correct for wave32 where the upper half is always empty.
4. **`may_use_sparse` whitelists only DKQ 512 and 576** (the DeepSeek shapes) - added 256/256.
5. **Tiling.** The third template arg of `switch_ncols1` is `ncols2`, and for RDNA with our
   `gqa_ratio = 12` the dispatch takes the `% 4 == 0` branch, so the sparse attempt happened at
   `(ncols1=1, ncols2=4)` - product 4. That does not compile: the AMD WMMA guard
   `ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 256 -> NO_DEVICE_CODE` means **RDNA has no FA device
   code at all below 16 tiles**, which surfaced as zero-length shared arrays plus two static
   asserts. The NVIDIA sparse tiling (1x8) therefore cannot exist on RDNA4 either.
6. **The valid AMD tiling (1x16) was never instantiated**, because `generate_cu_files.py` prunes
   `ncols2 in (16, 32)` for head sizes outside 192/320/576. Patched the generator with the reason
   and regenerated (one added `DECL_FATTN_MMA_F16_CASE(256, 256, 1, 16)`), plus an explicit RDNA
   request before the exact-divisor chain, since the sparse tiling is not what RDNA would pick.

## engagement threshold - and a blind spot in the E018 gate

`K->ne[1]` is the **current KV view length**, not the cache capacity, and the gate demands
`K->ne[1] >= max(4096, 2*n_kv_max)`. With `n_kv_max = 2051` that is **4102 KV**, so:

- the first two probe runs looked like "sparse does nothing" while actually being dense-only runs
  at too small a depth;
- **the golden fingerprint can never exercise sparse FA**: `tools/golden-corpus.md` is 3428 tokens
  and `llama-perplexity` needs `2 x ctx` tokens, so its context is capped at 1714 - far under 4102.
  `tools/sparse-corpus.md` + `-c 8192` is now the deep-context half of the gate;
- short-context decode is dense by design upstream, so any future "sparse didn't help my tg"
  observation must first check the depth.

The dense arm printed the gate 48 times at `-c 8192` while the sparse arm printed it 72: the RDNA
pre-dispatch has to ask `shall_use_sparse` before the tiling is chosen, so the call now happens
once per FA op build even when the answer is no. Host-side only, no measurable cost, but it is why
an unmodified dense run touches that function at all.

## regression this caught, and NVIDIA impact

`src/models/deepseek4.cpp:760` passes `n_kv_max > 0` **unconditionally**, so with the HIP guards
simply removed a DS4-family model on RDNA3/4 would dispatch to `case<576, 512, 1, 16>`, which has
no AMD device code (`DKQ > 256`) -> `NO_DEVICE_CODE` abort where today it computes dense. The arch
predicate is therefore `turing_mma_available(cc) || (amd_wmma_available(cc) && K->ne[0] <= 256)`.

NVIDIA behaviour is unchanged: its dispatch tail only ever requests `ncols2` in {8,4,2,1} for
DKQ <= 256, so the new whitelist entry is unreachable there. The only NVIDIA-facing effect is one
extra compiled instantiation (code size and CUDA build time), which upstream would want to hear
about before this became a PR.

## caveats

- Not tested on NVIDIA hardware (none here). MUSA still aborts, unchanged.
- The (1,16) tiling puts 12 heads into 16 slots, so 25% of the tile compute is wasted, and it
  still nets +23%. A ratio-12-capable tiling would need actual kernel work, not enablement.
- Sparse forces `nstages = 1` and disables `cp.async` (`fattn-mma-f16.cuh`), so on RDNA this path
  gives up the multi-stage prefetch the dense kernel uses. That it wins anyway is a statement about
  how much KV traffic the compaction removes.
- `Q4EXP_SPARSE_FA` is an experiment-branch scaffold. Upstream form would be to enable the model
  line directly, since `shall_use_sparse` already decides per-call whether sparse is legal.

## process notes

- `2>&1 > file` sends stderr to the *old* stdout, so the first two diagnostic runs looked like
  "no prints" when the prints were going to the terminal. Use `> file 2>&1`.
- `llama-bench` does not surface `GGML_LOG_WARN`, so engagement had to be proven with
  `llama-perplexity`. A run that silently does nothing looks identical to one that works.
- The `(1,4)` attempt failed at compile time with errors in `fattn-mma-f16.cuh` that do not name
  the config; `make` continued other TUs, so the first visible symptom was a *stale binary*
  reporting plausible numbers.

## the crash artifact nobody warns about

A HIP device exception (`NO_DEVICE_CODE` here, `HSA_STATUS_ERROR_EXCEPTION code: 0x1016`) makes the
runtime write a GPU core dump into the **current working directory** as `gpucore.<pid>`. On this
card each was **7.3 GB**, and two crashed runs put 14.6 GB of them in the repo root, untracked but
one `git add -A` away from disaster. Check `df` and `ls gpucore.*` after any HIP crash, and never
run the crashing binary from the repo root twice without looking.
