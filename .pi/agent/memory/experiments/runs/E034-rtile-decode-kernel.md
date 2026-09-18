# E034 - fattn-rtile: the RDNA3 decode kernel, wired up, and what sparse decode needs

T2 setup work on a kernel the user pulled from an RDNA3 experiment. Objective: get it to compile and
run on gfx1201 behind a flag, then make it read a QSA selection instead of a contiguous KV range, so
that decode can stop reading the whole cache. Why decode wants it: `plans/qsa-ple-in-prefill-and-decode.md`
section 3 - at 131k the decode step reads 3.2 GB of KV across 12 layers and the sparse selection would
cut that to 50 MB, worth roughly +50% on `tg` at that depth if the marginal cost really is bytes.

## What the user pulled, and what was wrong with it

| file | content | state as pulled |
|---|---|---|
| `fattn-rtile.cu` | the device kernel: `flash_attn_rtile<D, ncols2, type_KV>`, tile dataflow written for gfx1100, K/V staged through LDS in `nbatch_fa = 32` row tiles, q8_0/q4_0 consumed natively | complete, never compiled here |
| `fattn-rtile.cuh` | the host dispatch: `ggml_cuda_flash_attn_ext_rtile` and its type/gqa/ncols2 switches, plus `rtile_min_parallel_blocks` | **file names were swapped**: this is what the `.cu` should hold, and it `#include`d itself, and referenced `flash_attn_rtile` which is defined in the other file |
| `fattn-select.cuh` | new: `fattn_props` struct, extended kernel enum with RDNA ids, three hook declarations | complete |
| `fattn-rdna.cu` | the RDNA hook implementations | selection hard-wired to `return BEST_FATTN_KERNEL_DEFAULT` below a FIXME, `can_use_rtile` tested only `GGML_CUDA_CC_IS_RDNA3`, and it called `ggml_cuda_flash_attn_ext_tile_allmma`, whose file is not in the tree |
| `fattn.cu` | +9 lines: call the RDNA hook after `gqa_ratio_eff` is derived | did not include `fattn-select.cuh`, still defined its own copy of the enum, and neither the dispatch switch nor the alloc-size switch knew about the RDNA ids |

## What I changed to get it building

1. Swapped the contents of `fattn-rtile.cu` / `.cuh` to match upstream's `fattn-tile` convention
   (`.cuh` = kernel + declaration, `.cu` = dispatch), and appended the declaration
   `void ggml_cuda_flash_attn_ext_rtile(...)` to the header so `fattn-rdna.cu` can call it.
2. `launch_fattn` gained a trailing `const int min_parallel_blocks = 1`, applied as
   `parallel_blocks = min(ntiles_KV, max(parallel_blocks, min_parallel_blocks))` before the wave
   efficiency loop. rtile already called `launch_fattn` with twelve arguments, passing its KV-split
   floor as the last one; the current signature had eleven, so the file could not compile. Default 1
   leaves every existing caller bit-identical.
3. `fattn.cu`: include the header, drop the duplicated enum, and route the two RDNA ids through the
   hooks in both the alloc-size switch (`ggml_cuda_fattn_need_f16_rdna`, which is what keeps q8_0/q4_0
   KV from being dequantized) and the dispatch switch. The dispatch switch needed the selected kernel
   in a named variable first.
4. `fattn-rdna.cu`: `GGML_CUDA_CC_IS_RDNA4` accepted next to RDNA3; the arm gated on
   `GGML_FATTN_RDNA_RTILE` (unset = off, so nothing changes for anyone); `tile_allmma` now aborts with
   a named message instead of calling a function that is not declared; and the FIXME's crash log moved
   here, where it belongs.

Full build clean on gfx1201, ROCm 7.1.1.

## The crash that stopped them, and what it actually means

Their FIXME carried this, which is the whole reason the arm was disabled:

```
GGML_ASSERT(n_kv_max > 0) failed        at ggml/src/ggml-cuda/fattn-common.cuh:1099
#6  launch_fattn<256, 64, 2>(ggml_backend_cuda_context&, ggml_tensor*, ...)
#7  ggml_cuda_flash_attn_ext_tile_allmma(...)
#8  ggml_cuda_graph_evaluate_and_capture(...)      #12 llama_context::graph_compute
#13 llama_context::process_ubatch                   #14 llama_context::decode
#16 test_prompt(...)                                #17 llama_bench(...)
```

`use_sparse` is a positional `bool` in `launch_fattn`, immediately before `warp_size`, and
`fattn-common.cuh` asserts `n_kv_max > 0` whenever it is true. A node built without the QSA hint has
`n_kv_max == 0`, so passing the flag positionally - which is what `tile_allmma` did - aborts the first
time a non-QSA model decodes. Not a memory bug, and not specific to their kernel: it is the price of
the parameter being positional. rtile takes the same risk the moment its `false` becomes a `true`.

## The contract for the sparse version, from the code that already exists

`launch_fattn` does nearly all of it when handed `use_sparse = true`:

- `n_kv = use_sparse ? n_kv_max : K->ne[1]` at `:1131`, so `ntiles_KV`, `parallel_blocks` and the grid
  are sized for the selection, not the cache. Their `min_parallel_blocks` floor is a no-op there.
- `KV_max.alloc(n_kv_max * mask_rows)` then `ggml_cuda_flash_attn_ext_compact_mask(mask, KV_max.ptr,
  n_kv_max, stream)` at `:1095-1102`: the same scratch buffer that holds per-tile KV lengths in the
  dense case holds an `int32` index row per (sequence, query) in the sparse case.
- Padding is a sentinel, not a length: `fattn.cu:91-92` writes `-1` for the entries past the count.
- The mma kernel's idiom, worth copying verbatim (`fattn-mma-f16.cuh:1798-1799`):
  `KV_max = use_sparse ? KV_max_ptr : nullptr` becomes the reverse pair, so one pointer argument
  serves both readings and `nullptr` selects the mode in the inner loop.

For rtile the change is one loop and one helper. `fattn-rtile.cu:374` reads the length, and `:379` is
`for (int k0 = blockIdx.y*nbatch_fa; k0 < k_VKQ_max; k0 += gridDim.y*nbatch_fa)`; `:406` and `:408`
stage `nbatch_fa` rows starting at `k0`, and `:472` reads `maskh[k0 + threadIdx.x]`. Sparse means the
tile's 32 rows come from `indices[tile_base + 0..31]`, so `rtile_stage_tile` needs a per-row address
form, and the mask read becomes `maskh[idx]` with `idx < 0` skipped. The index row is per (sequence,
query) and in decode there is exactly one query per sequence, so every one of the 12 q heads in the
tile walks the same list, which is what makes this cheap here. Because `KV_max_ptr` carries two different meanings, the mode has to reach the kernel as a
template parameter, not as a runtime test on that pointer.

## Not yet established

- whether rtile is faster than the vec kernel on gfx1201 at this shape at all - that is the dense arm,
  and the two claims (better kernel, better because sparse) must not be measured in one step;
- the per-device `gqa_ratio` under `-sm tensor` on the bench, which decides whether the
  `gqa_ratio % 2 == 0` condition holds: 24 q heads and 2 kv heads over 4 cards could come out 3, and
  then the arm must not be selected - `can_use_rtile` checks it, but it should be printed once;
- whether their `tile_allmma` kernel should come back into the tree at all;
- and the missing design doc `.pi/local/fattn-rtile-design-2026-08-07.md`, whose sections 24-25 hold
  the reasoning behind the per-cell parallel-block floors (q4_0 144, q8_0 144/96/48, f16 144).
