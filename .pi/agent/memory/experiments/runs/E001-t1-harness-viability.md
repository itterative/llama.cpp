# E001 - is the T1 harness usable on ROCm0

- date: 2026-09-17
- machine: dev-rx9070-16g
- tier: T1
- status: blocked
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
