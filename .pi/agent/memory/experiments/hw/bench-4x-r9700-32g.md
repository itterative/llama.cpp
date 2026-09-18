# hw profile: bench-4x-r9700-32g (v1)

The 4-GPU box. The only machine that can run real `qwen4exp`, so the sole source of T2
evidence. User-supplied facts marked (user); everything else still to confirm.

## Identity

| field | value |
|---|---|
| cards | 4x **AMD Radeon AI PRO R9700**, **32 GB** each (user) |
| gfx target | **gfx1201** (user-confirmed), RDNA4 / Navi 48 XT class |
| total VRAM | 128 GB |
| **system RAM** | **62.7 GiB total** (screenshot 2026-09-18 19:21, `gtop` mem panel; 7.70 GiB used, 12%) - the "32g" in this profile's name is **not** RAM, and the earlier reasoning that a 30-36 GB PLE table cannot fit in RAM was wrong |
| role | T2: real `qwen4exp` end-to-end, multi-GPU |
| reachability | separate machine, not reachable from this session; user runs commands by hand |

Still unconfirmed: ROCm version, driver, PCIe topology / any Infinity Fabric links, CPU +
system RAM, OS.

**gfx1201 on both boxes, so they share an ISA target.** The dev build has no
`AMDGPU_TARGETS` override and compiled gfx1201-only, which means a dev-box build is
ISA-valid on the bench box. It does *not* mean a dev-box binary is a valid measurement
there: the `LD_LIBRARY_PATH` shadowing caveat applies independently, and ROCm version
differences still make numbers non-comparable. Cross-shipping a binary is allowed; using
it as a baseline partner is not.

## Deployment as the user actually runs it (user-supplied)

| item | placement |
|---|---|
| main weights | **Q4_K_M**, split across the 4 GPUs, **~80 GB** total |
| PLE n-gram table (`per_layer_token_embd.weight`) | **Q5**, kept in **system RAM**, not on GPU |
| vision tower (`mmproj`) | also in **RAM**, "so I can run at full context" |
| context | full context (config says 262,144) |

Matches the arithmetic: non-PLE is ~129 B params, which at Q4_K_M is ~72-80 GB; the PLE
table is ~51 B params, which at Q5 is ~30-36 GB of host RAM.

128 GB of VRAM against ~80 GB of weights leaves real headroom for KV and recurrent state -
so on this box the question is not "does it fit" but "what is the bottleneck". That is a
much better position than the one I sketched from param counts alone, and it retires the
feasibility panic in the earlier draft of `plans/model-shape.md`.

## One thing that may not match your mental model

The n-gram table is described as "in RAM". With defaults it may not be *resident*:

`-lzm/--lazy-mode` defaults to `auto`, which enables lazy reading for **any tensor larger
than 4 GiB**, name-agnostic (`src/llama-model-loader.cpp:1088-1100`; `auto_min_size` at
`:1093`). A ~30-36 GB Q5 table qualifies, and lazy means "read the rows of such tensors
from disk on demand instead of keeping them resident (requires mmap)"
(`common/arg.cpp:2707-2710`).

Consequence: the table is served through the page cache, so a random gather over ~20 M rows
faults pages in as they are touched. RSS creeps toward the full size with use and *can* be
evicted under memory pressure - which presents as irregular `tg` latency rather than a
steady number. That is a measurement confounder for this whole project, not trivia.

`LLAMA_LAZY_MODE_AUTO` falls back to `OFF` if any selected device reports
`mmap_support == false` (`src/llama-model.cpp:1455-1465`), so behaviour also depends on
`--load-mode`.

### Check on the bench box

Cheap, and needs no rerun - it is already in a load log:

```sh
grep -iE "lazy read enabled|per_layer_token_embd" <the run's log>
```

If that tensor reports `lazy read enabled`, the A/B set is `-lzm off` (fully resident) vs
`-lzm auto` (today) vs `--load-mode mmap+mlock`, with `free -h` / RSS before and after.
Logged as L1 in `plans/backlog.md`.

## Characterisation block

Paste on the bench box, save to `results/hw-bench-4x-r9700-32g-v1.txt`, bring back.

```sh
{
  echo "### date"; date -u
  echo "### os"; cat /etc/os-release
  echo "### rocm"; ls -d /opt/rocm* 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null; rpm -q rocm-core hipcc 2>/dev/null || dpkg -l | grep -E "rocm-core|hipcc"
  echo "### asic"; timeout 30 amd-smi static --asic 2>&1 | grep -E "MARKET_NAME|TARGET_GRAPHICS_VERSION|VRAM_TOTAL|NUM_COMPUTE_UNITS|DEVICE_ID"
  echo "### topology"; timeout 30 amd-smi topology 2>&1 | head -30
  echo "### lspci"; timeout 30 lspci | grep -i -E "vga|display|3d|bridge"
  echo "### cpu/ram"; grep -m1 "model name" /proc/cpuinfo; nproc; free -g | head -2
  echo "### hugepages"; grep -i huge /proc/meminfo | head -6
  echo "### llama devices"; LD_LIBRARY_PATH="$PWD/build/bin" ./build/bin/llama-cli --list-devices
} > hw-bench-4x-r9700-32g-v1.txt 2>&1
```

`--list-devices` should print four `ROCmN` lines totalling ~128 GiB; that is the fastest
confirmation the box sees all four cards and that its build matches their gfx target.

## Known constraint inherited from the code

`-sm tensor` is **not available** for qwen4exp (`llm_arch_supports_sm_tensor` returns
false, `src/llama-arch.cpp:1161`, upstream `// TODO: fix test-llama-archs`), so distributing
the ~80 GB of GPU weights is limited to `-sm layer`. With 12 full-attention layers among 36
linear ones at interval 4, plus the one PLE layer and 512-expert MoE blocks, per-card
balance across 4 cards is a live question (backlog H1).
