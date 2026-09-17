# E001 - is the T1 harness usable on ROCm0

- date: 2026-09-17
- machine: dev-rx9070-16g
- tier: T1
- status: done
- parent: -
- commit: `ebbb185227c31f1652f1445e2623563d2f67fe5a` (tree: clean; only untracked `.pi/agent/memory/`)
- build: pre-existing `build/`, `GGML_HIP=ON`, `BUILD_SHARED_LIBS=ON`, no `AMDGPU_TARGETS` override
- model: none (harness only)

## hypothesis

The existing low-VRAM harness in-tree - `test-llama-archs` synthetic models, `test-fusion`
counts, `test-backend-ops`, `llama-bench` - runs on the HIP backend under gfx1201, which
would make real T1 experimentation possible without the 16 GB weight problem.

## prediction

All four tools run to completion against `ROCm0`. Falsified by any one of them failing
for a reason that is not a usage error on my side - because then the fix is a
prerequisite, not a nuisance, and no later experiment can gate correctness.

## conditions

- no `LD_LIBRARY_PATH` pin on the first attempts (this was the bug found)
- `rocm-smi --showmeminfo vram` before: 59,957,248 B used of 17,095,983,104 (idle)
- no other GPU user; box otherwise idle
- ROCm 6.4.4 / hipcc 19-14.rocm6.4.2

## results

| probe | command | outcome |
|---|---|---|
| device naming | `llama-cli --list-devices` | `ROCm0: AMD Radeon RX 9070 (16304 MiB, 16238 MiB free)` - so `--device ROCm0`, not `CUDA0`/`HIP0` |
| device init banner | any ggml-cuda binary | `Device 0: AMD Radeon RX 9070, gfx1201 (0x1201), VMM: no, Wave Size: 32` |
| op support probe | `test-backend-ops support -b ROCm0 -o MUL_MAT` | **died**: `symbol lookup error: undefined symbol: ggml_dsv4_hc_pre_gated` |
| op support probe | `test-backend-ops support -b ROCm -o ADD` | ran, but matched no device and skipped both backends: `-b` is an exact `strcmp` against `ggml_backend_dev_name()`, so the filter must be `ROCm0` |
| synthetic models | `test-llama-archs -o /tmp/dummy-models` | **0 files produced**; `failed to create llama model`, `key llama.attention.causal has wrong type f32 but expected type bool` |
| stale lib inspection | `nm -D ~/.local/lib64/libggml.so.0.24.0 \| grep -c dsv4_hc` | `0` -> the installed lib predates `37b53fd45` (hc ops), installed Sep 15 |
| resolution, unpinned | `ldd build/bin/test-backend-ops` | `libggml.so.0 => /home/sd/.local/lib64/...` (**wrong**) |
| resolution, pinned | `LD_LIBRARY_PATH=$PWD/build/bin ldd ...` | `libggml.so.0 => .../polished-quarry/llama.cpp/build/bin/...` (correct) |

## raw

Nothing worth keeping: the two failures are reproduced by the commands above. The
`ldd`/`nm` outputs are summarised inline since they are one-liners.

## verdict

**blocked** - hypothesis not yet tested, because both failures were confounded by the
loader, not by the backend. Established:

1. The device name is `ROCm0` and the `-b` filter is an exact match. Worth knowing: a
   wrong filter is silent and reports "N/N backends passed / OK", which is exactly the
   shape of failure that produces a confident bogus result later.
2. `~/.local/lib64` on this box shadows the working tree for every dynamic binary. With
   `BUILD_SHARED_LIBS=ON` and no rpath there is nothing forcing the correct lib. Any
   T1 number taken without the pin is measuring code from Sep 15.
3. `test-llama-archs` failing under a mismatched `libllama` explains the `causal`
   type error - the loader was old, the key writer was new. Not evidence of a qwen4exp bug.

Still unproven: whether the HIP backend can drive the synthetic qwen4exp model at all.

Not yet even attempted: the reported `test-backend-ops` crash on AMD. The one crash seen
here was a missing symbol, i.e. a different failure, so the AMD crash is still an open
item and it is the gate for rule 3 (correctness before perf).

## notes

Rerun with the pin before drawing anything from the two failures:

```sh
cd /home/sd/Repos/worktrees/llama.cpp/polished-quarry/llama.cpp
export LD_LIBRARY_PATH=$PWD/build/bin
rm -rf /tmp/dummy-models && mkdir -p /tmp/dummy-models
./build/bin/test-llama-archs -o /tmp/dummy-models
ls -la /tmp/dummy-models | grep qwen4exp
./build/bin/test-fusion --models /tmp/dummy-models --device ROCm0 --record /tmp/f.csv
```

Then decide the correctness gate. If `test-backend-ops` really crashes on HIP, the
fallback is a fixed-prompt logprob diff (`scripts/compare-logprobs.py`) against CPU on
the synthetic model - slower per-op but it exercises the same kernels end to end.

Structural thought this raises: `BUILD_SHARED_LIBS=ON` plus a shadowing `~/.local` is a
permanently unsafe default for a perf branch. A static build, or an rpath, removes the
whole class of error rather than relying on remembering an export. Candidate backlog B0.
(Note: `GGML_STATIC` is a hard `FATAL_ERROR` on the HIP path, so for B0 it is rpath or
nothing.)

---

## Update, same day - harness run with the pin (user authorised GPU use)

`status: blocked -> done`. The user recalled `test-backend-ops` hanging the GPU, but
specifically **on the flash-attention tests, and on the bench box**, with the mul_mat tests
fine. Re-tested here under the pin, in widening stages:

| stage | scope | result |
|---|---|---|
| 1 | `support`, 13 ops incl. HC/GDN/TOP_K/SET_ROWS/ARGSORT | **rc=0**, 9906 supported / 2384 unsupported cases |
| 2 | `test`, the qwen4exp op set *excluding* FLASH_ATTN_EXT | **1500/1500 passed** |
| 3 | `test -o FLASH_ATTN_EXT -p n_kv_max=512` (the sparse cases) | **11/11 passed** - no abort, see below |
| 4 | `test -o FLASH_ATTN_EXT`, whole suite | **3973/3979 passed, rc=1**, no hang, no crash |

**The AMD `test-backend-ops` blocker is removed for the dev box.** Evidence:
`results/E001-harness-summary.txt` (per-op support tallies, the six failing cases, the
commands) plus `results/E001-test-fa-sparse.txt` (the retained 2.9 KB evidence for the
"sparse cases pass by computing dense" finding). The three large raw logs were distilled away
rather than committed, per `PROTOCOL.md` 8 - they regenerate from the commands in the summary.

### The 6 failures

All are `hsk=192, hsv=128`, gqa 8 or 16, with permuted K/V views; error 0.0033-0.0298
against a 0.0005 tolerance. Numerically wrong, not a crash. **qwen4exp is 256/256, and that
set passed 142/142**, so this is not our path - recorded in `plans/backlog.md` as an
observed-but-out-of-scope defect rather than chased.

### Why the sparse cases did not hit the GGML_ABORT

The 18 nonzero-`n_kv_max` FA cases all report `SUPPORTED` on `ROCm0` (they select a kernel
via `ggml_cuda_flash_attn_ext_supported` -> `get_best_fattn_kernel != NONE`, which never
consults `n_kv_max`), and then **pass**, because `use_sparse` is false on HIP so
`compact_mask` is never called and the case silently computes dense over the same mask.

This corrects a claim in `rdna4-rocm-build`: the `GGML_ABORT` is unreachable, not a
landmine, and the implication for the model is sharper than I put it -

> **flipping `n_kv_max` at `src/models/qwen4exp.cpp:767` is inert on ROCm.** It changes
> neither results nor speed, because the sparse branch is compiled out (`#if
> !defined(GGML_USE_HIP)` at `fattn.cu:133-140`) rather than merely gated off. H4b is
> therefore real kernel work in `mma_f16`, not a flag flip, and H4a cannot be approximated
> by toggling `n_kv_max` either.

### Still open

- **The dummy-model leg of the original hypothesis was never re-run.** E001 set out to prove
  `test-llama-archs` + `test-fusion` + `llama-bench` work too; what got re-tested under the
  pin is `test-backend-ops` only. Those three remain unverified and are E002's first
  commands, so E001 is closed on the blocker it actually gates (B1, the correctness gate)
  and not on its full original scope.
- The bench box hang is unreproduced and now has to be explained differently: not a general
  AMD problem. Candidates are the 4-GPU config, a different ROCm version, or a different code
  state - all three are B2 unknowns. If it recurs there, capture which stage dies.

---

## Update 2 - the rest of the harness, pinned

Closes E001's residual scope. One of the three legs passed, one is structurally impossible
on this backend, and the smoke run produced a trap worth writing down.

| leg | result |
|---|---|
| `test-llama-archs -o /tmp/dummy-models` | **rc=0, 111 models, no errors.** Confirms the original 0-files failure was purely the loader trap. `qwen4exp-moe.gguf` = 19.24 MB, 4.80 M params, all F32, 2 layers |
| `test-fusion --model .../qwen4exp-moe.gguf --device ROCm0` | **cannot run on this backend at all**: `device 'ROCm0' does not export the generic fusion debugging API (ggml_backend_fusion_*)`. Only `ggml/src/ggml-metal/ggml-metal.cpp` implements it `[v]`, which is why `tests/fusion/` contains exactly one CSV (MTL). Closes P3 as not-doable and puts the N1 IMROPE-fusion gap back to needing a different instrument |
| qwen4exp graph on GPU | **executes end to end.** `llama-bench -ngl 99 -p 32 -n 8` -> rc=0; also rc=0 with `-fa 1`, so `FLASH_ATTN_EXT` runs on this arch's real graph. Load log: `offloaded 3/3 layers to GPU`, `layer 0/1/2 assigned to device ROCm0`. Not a trivial outcome given `36b101543` fixed a "cuda abort" for this model |

Trap: **do not feed a dummy model to a tool that tokenizes.** `llama-cli` aborted with
`src/llama-vocab.cpp:3393: GGML_ASSERT(tokenizer && "Tokenizer not initialized...")`,
backtrace through `tokenize_input_prompts` - i.e. a vocabulary failure, nothing to do with
the backend. Dummy GGUFs are for token-id consumers (`llama-bench`,
`test-save-load-state`, which is what `tests/CMakeLists.txt:247` uses them for).

Also: `-no-conversation`/`--no-conversation` was rejected by `llama-cli` here; `-st` is the
flag for a non-interactive run.

### What this does *not* establish

The t/s numbers (`pp32` 11768, `tg8` 325) are not a baseline and are not recorded as one -
see E002. A 4.8 M-param 2-layer model in F32 fits in cache, so its timings measure launch
and input-path overhead, not the real bottleneck. Chasing graph splits from that number was
dropped: `GGML_SCHED_DEBUG` only calls `ggml_backend_sched_print_assignments`, and llama-bench
didn't surface useful output - an unproductive thread, noted so it is not retried blindly.
