# hw profile: bench-4xr9700 (v0 - NOT YET CHARACTERISED)

The 4-GPU RDNA4 box. The only machine that can run real `qwen4exp` weights, so the
sole source of T2 evidence. Nothing in this file is verified yet except what the user
stated; treat every field as a question, not a fact.

## Stated

- 4x AMD RDNA4 cards, user's shorthand "r9700".
- Separate machine from the dev box; not reachable from this session.
- Backend of interest: ROCm/HIP.

## Must confirm before the first T2 number

| field | why it matters | value |
|---|---|---|
| exact SKU + `TARGET_GRAPHICS_VERSION` (gfx1200 vs gfx1201 vs ...) | the dev box is gfx1201. A different gfx means different ISA, different MMQ/MMVQ tuning, and a dev-box build that will not even load | ? |
| VRAM per card, and total | decides quant size, n_ctx, whether the model fits at all without offload | ? |
| ROCm version + install source (amd repo vs Fedora vs offline) | ggml-cuda gates behaviour on ROCm version; 6.4 vs 7.2 are different code paths | ? |
| kernel driver + amdgpu firmware | RDNA4 support moved fast; affects peer access and graphs | ? |
| PCIe topology (`amd-smi topology` or `lspci -tv`) | 4x on one root complex means all inter-GPU traffic shares x16 host links; determines whether layer split is bandwidth-bottlenecked | ? |
| any Infinity Fabric / peer links | if present, `-sm layer` with peer access behaves very differently | ? |
| CPU + RAM, and NVMe | the loader/mmap path and prompt-read times show up in server-side latency numbers | ? |
| OS + glibc | cross-box binary shipping (see below) | ? |
| can it reach HuggingFace | if not, weights ship out-of-band and the record must note where they came from | ? |

## Binary shipping

Do not copy `build/bin/*` from the dev box and call it the same measurement. The dev
build auto-targets the local gfx1201 only (no `AMDGPU_TARGETS` in the cache), so on a
different gfx it fails or silently degrades; and this tree is `BUILD_SHARED_LIBS=ON`, so
the `.so` set has to travel too and lands in whatever `LD_LIBRARY_PATH` the box already
has.

Preferred: build on the bench box, and record its `cmake` cache + commit sha in the run
record. If a dev-box binary must be reused, the record has to say so explicitly and the
first job is proving `ldd` resolves to the shipped libs.

## One-shot characterisation block

Paste this on the bench box, save the output as `results/hw-bench-4xr9700-v1.txt`, and
bring it back. It fills every field above except OS details.

```sh
{
  echo "### date"; date -u
  echo "### os"; cat /etc/os-release
  echo "### rocm"; ls -d /opt/rocm* 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null; rpm -q rocm-core hipcc 2>/dev/null || dpkg -l | grep -E "rocm-core|hipcc"
  echo "### asic"; timeout 30 amd-smi static --asic 2>&1 | grep -E "MARKET_NAME|TARGET_GRAPHICS_VERSION|VRAM_TOTAL|NUM_COMPUTE_UNITS|DEVICE_ID"
  echo "### topology"; timeout 30 amd-smi topology 2>&1 | head -30
  echo "### lspci"; timeout 30 lspci | grep -i -E "vga|display|3d|bridge"
  echo "### cpu"; grep -m1 "model name" /proc/cpuinfo; nproc
  echo "### mem"; free -g | head -2
  echo "### llama devices"; LD_LIBRARY_PATH="$PWD/build/bin" ./build/bin/llama-cli --list-devices
} > hw-bench-4xr9700-v1.txt 2>&1
```

## Notes for whoever runs it

- `--list-devices` needs the build present on that box, hence the inline
  `LD_LIBRARY_PATH`. If the box has its own `~/.local/lib64` llama.cpp install, it will
  shadow the build exactly like the dev box does - check with `ldd` first.
- VRAM per card from `--list-devices` is the number to trust for capacity planning; the
  smi tools report it in bytes and in different units across ROCm versions.
- if `amd-smi` is unavailable, `rocm-smi --showhw` and `rocminfo | grep -E "Name|Marketing"`
  give the same gfx string.
