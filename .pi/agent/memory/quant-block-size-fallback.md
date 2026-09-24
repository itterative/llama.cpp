---
name: quant-block-size-fallback
description: MoE/linear widths not divisible by a k-quant's 256 block silently bump tensors to a bigger type at quantize time; how to detect it and the --tensor-type lever
category: performance
keep_updated: true
---

# Quantize block-size fallback: a narrow tensor silently costs 33% more bytes

`src/llama-quant.cpp:372-408` (`tensor_type_fallback`): if `ncols % ggml_blck_size(target) != 0` the
tensor cannot use the planned type and is demoted - for a Q4_K target the fallback is **Q5_0**, which is
6.0 bpw against Q4_K's 4.5. The fallback prints only under `llama-quantize`'s own verbose line
(`-> falling back to %s`), and a downloaded GGUF carries no trace of why a tensor has its type.

The case that matters for a 10-of-512 MoE: expert `ffn_down` has `ncols = moe_intermediate_size`. Any
value that is not a multiple of 256 pushes **every** expert down tensor up a class. With 640 (this model:
48 layers x 512 experts = 24,576 tensors) `640 % 256 = 128`, so all of them land on Q5_0 - a third of the
expert parameters carrying 33% more bytes than the recipe intended, and through a 32-element block, which
costs more scale metadata and more dequant work per element than a 256-block type.

`gate`/`up` are unaffected because their `ncols` is `n_embd` (2560, divisible by 256). Same trap applies to
any model with a narrow intermediate or a head-dim-ish width: check `ncols % 256` before concluding that an
M-recipe produced uniform Q4_K experts.

## Detecting it in a file you did not quantize

- The loader prints a per-type census at INFO level: `src/llama-model-loader.cpp:825`,
  `- type  q5_0:  N tensors`. A count near `n_layer * n_expert` is the signature.
- `llama-gguf <f>.gguf r` prints tensor names and byte sizes (not types). Dividing size by `ne[0]*ne[1]*...`
  gives bpw directly, so `blk.7.ffn_down.3.weight` at 6.0 bpw vs `blk.7.ffn_gate.3.weight` at 4.5 confirms
  it without any extra tool.
- Enum trap: slots 4 and 5 are the *removed* Q4_2/Q4_3, so numeric type 6 is **Q5_0** and 7 is Q5_1. A
  profiler row reading `(ggml_type)6` is Q5_0. Misreading this sent me down the wrong path once.

## The lever

`llama-quantize --tensor-type '<regex>=<type>'` (parsed at `src/llama-quant.cpp:17-19`, applied via the
pattern list at `:188-196`) can name the tensors explicitly. `q4_0` is a 32-block type at 4.5 bpw, so it is
valid at `ncols = 640` and removes the 33% byte penalty; the cost is accuracy on the down projections,
which `llama-quantize`'s per-tensor quantization analysis can price before committing to the file. It is a
file-level change: no kernel work, and the same byte saving lands on per-card VRAM under `-sm tensor`.

Before doing it, get the sign from the model's own PPL - this is a quality-relevant change, not free.
