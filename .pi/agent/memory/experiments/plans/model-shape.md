# Model shape - real qwen4exp, vs the synthetic dummy

Sources, fetched 2026-09-17 as **text only, no weight shards**:

- `config.json` from `Qwen/Qwen3.8-Flash-Next` (4745 B) - authoritative for all dims.
- Hub API `?expand=safetensors` -> `{"BF16": 179,999,981,424, "I64": 35}`, `total
  179,999,981,459` params; `metadata.total_size` = 359,999,963,128 B = **360 GB in bf16**.
- `model.safetensors.index.json` (170 KB) - 1658 tensors, 131 shards.
- Local code reading, cited inline.

Raw copies: `results/hf-config.json`, `results/hf-index.json`, `results/hf-api-meta.json`,
`results/hf-generation-config.json`. Exact requests, all text, no weight shards:

```sh
curl -sL https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json
curl -sL https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/model.safetensors.index.json
curl -sL "https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next?expand=safetensors"
curl -sL https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/generation_config.json
```

## The headline: this is a 180 B MoE whose largest object is one hash-gathered table

| | params | bf16 GB | share |
|---|---|---|---|
| MoE experts (48 layers x 512 experts x 3 x 2560x640) | ~120.8 B | ~242 | 67% |
| **PLE n-gram table** (one layer) | **~51.2 B** | **~102** | **28%** |
| token emb + output (untied, 248320 x 2560 x 2) | ~1.3 B | ~2.5 | 1% |
| attention (12 layers) + GDN (36 layers) + HC + routers + vision | remainder | ~13 | 4% |

The PLE row is an inference, not a Hub-reported number, but two independent routes agree:
`ngram_vocab_size_base = 20,000,000` x `ple_embed_dim = 2560` = 51.2 B, and 51.2 B is what
the 180 B total leaves over the expert stack. `split_ngram_parts = 128` matches the 128
`...ple.ple_embedding.ngram_embedding.shard_N.weight` tensors in the index, and the
converter comment says plainly "**the shards concatenate into a tensor of well over
100 GB**" (`conversion/qwen4exp.py:142`).

## Real config (text_config), and the dummy value beside it

| key | real | dummy | why it matters |
|---|---|---|---|
| `num_hidden_layers` | **48** | 2 | |
| `full_attention_interval` / `layer_types` | 4; 12 of 48 full, 36 linear | 2 layers, ratio on all | only 25% of layers can benefit from QSA at all |
| `hidden_size` | 2560 | 256 | |
| `head_dim` | **256** | 128 | `can_use_vector_kernel` is `<= 256 && % 64 == 0 && != 192` (`ggml/src/ggml-cuda/fattn.cu:611`) -> 256 squeaks in at the boundary. Gates H4b |
| `num_attention_heads` / `num_key_value_heads` | 24 / 2 | 2 / - | GQA ratio 12 |
| `partial_rotary_factor` | **0.25** | ~0.5 | n_rot = 64 of 256; most of the head is *not* rotated |
| `mrope_section` / interleaved | [11,11,10], true | 4 equal sections | rope-fusion experiments must match this |
| `indexer_n_heads` / `indexer_head_dim` / `indexer_kv_heads` | 4 / 128 / 1 | 64 / 128 / - | the dummy invents 64 indexer heads; real is 4 |
| `indexer_budget` (top_k) / `indexer_compress_ratio` | **2048** / 4 | 8 / 4 | 2048 tokens = 512 blocks + tail. At 256 k context, sparse-vs-dense is ~100x on attention work - which is what makes H4b worth porting |
| `max_position_embeddings` | **262,144** | 256 | the dummy cannot express the regime that matters |
| `hc_count` / `hc_lowrank` | **4** / 320 | 4 / 8 | hc_count matches the dummy, and matches Vulkan's `ne[1] == 4` fused-path gate (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:15319`). So the dummy does *not* mislead on HC shape here |
| `num_experts` / `num_experts_per_tok` | **512** / 10 | none | 10/512 = 2% of expert weights touched per token per layer: `tg` is a random-read problem, not a flops problem |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 | n_ff 384 | experts are narrow (640) and numerous (512) |
| GDN: `linear_num_key_heads` / `linear_key_head_dim` / `linear_num_value_heads` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 16 / 128 / 48 / 128 / 4 | ssm_inner 256 | key 2048 wide, **value 6144 wide** - the recurrent state is big |
| `mamba_ssm_dtype` | **float32** | - | recurrent state in fp32, x36 linear layers: a real VRAM line item, and it resists the fp16 tricks |
| `ngram_size` / `heads_per_ngram` | 3 / 8 | 3 / 2 | `ple_n_heads = (3-1)*8 = 16` gathers per token, vs 4 in the dummy |
| `ple_layer_ids` | **[2]** (1-based -> layer index 1) | {0} | the 102 GB table serves exactly one layer |
| `vocab_size` / tie | 248320 / untied | 128 | |
| `rope_theta` | 1e7 | - | |
| multimodal | `Qwen4ExpForConditionalGeneration`, vision tower depth 27, hidden 1152, patch 16; image/video token ids | - | an mtmd path exists; text-only runs still carry the tower |
| MTP | 1 hybrid full-attention layer in config, **but the exporter drops it** (`no_mtp`, `supports_mtp_export = False`, `conversion/qwen4exp.py:21-22`) | - | no MTP draft head in the GGUF: speculative decoding via MTP is off the table for now |

## Deployment (user-supplied, 2026-09-17) - feasibility is already solved

On the bench box (4x Radeon AI PRO R9700, 32 GB each = 128 GB, RDNA4):

| component | placement | approx size |
|---|---|---|
| main weights | **Q4_K_M**, spread over the 4 GPUs | **~80 GB** (user) |
| PLE n-gram table | **Q5**, in **system RAM** | ~30-36 GB |
| vision tower / `mmproj` | in **RAM** | - |
| context | full, i.e. the 262,144 the config advertises | - |

Corroborated by arithmetic, which is the useful check: non-PLE params ~129 B at Q4_K_M
(~0.56 B/param) is ~72 GB, so "~80 GB" is the right order; 51.2 B PLE params at Q5 is
~33 GB of host RAM. Earlier drafts of this file guessed 4x16 GB and concluded the model
might not fit; that was wrong in both hardware and conclusion, and is kept here only as a
note that **the open question on this box is bottleneck identity, not capacity**.

Two follow-ups that this raises directly:

- with ~80 GB of weights and 128 GB of VRAM, `-sm layer` imbalance across 4 cards is a
  tuning knob rather than a hard wall (backlog H1)
- the Q5 table in RAM may be **lazy-read rather than resident** by default: `-lzm auto`
  applies to any tensor over 4 GiB (`src/llama-model-loader.cpp:1093`). See
  `hw/bench-4x-r9700-32g.md` and backlog L1 - this is the most likely source of irregular
  `tg` on a 20 M-row random gather.

`per_layer_token_embd.weight` is category `TOKEN_EMBD` (`src/llama-quant.cpp:104-107`), so
it follows `--token-embedding-type` and can be pinned individually with `--tensor-type`
(`src/llama-quant.cpp:686-700`). That is the lever that decides whether this model is
runnable at all on the bench box, and it is why P5/B2 (per-card VRAM) is now blocking
rather than merely useful.

## Where the dummy model will mislead, ranked

1. **`max_position_embeddings` 256 vs 262,144.** Every context-scaling effect - the QSA
   tax in E003, mask traffic, KV growth - is invisible at 256. E003 must override `-c`,
   and should say so in its record.
2. **PLE 4 heads x 128 rows vs 16 heads x ~20M rows.** The dummy's table fits in L2.
   Any statement about PLE cost, VRAM, or locality from the dummy is worthless.
3. **`head_dim` 128 vs 256.** Different FA kernel family eligibility.
4. **`indexer_n_heads` 64 vs 4.** The dummy spends *more* on indexing than the real
   model does - so E003's indexer-cost share is an overestimate.
5. `indexer_budget` 8 vs 2048: the dummy's sparse selection keeps 8 of 256 positions,
   i.e. it selects *less* than the real budget selects relative to context. Direction
   of the effect is right, magnitude is not.
6. `compress_ratio` 4 on **all** layers in the dummy (`tests/test-llama-archs.cpp:262`)
   vs 0 on linear layers in reality (`conversion/qwen4exp.py:60-66`). The dummy exercises
   a configuration the exporter never produces, and under-covers the real `ratio == 0`
   path.
7. 2 layers vs 48: fixed per-run costs dominate a 2-layer graph, so per-layer overheads
   look smaller than they are.

## What the dummy is still genuinely good for

Op-graph shape and which ops exist; whether an op runs on the GPU or falls back to CPU;
fusion-pattern counts (`test-fusion`); correctness of a kernel against a CPU reference;
graph-split counts; and any change whose effect is per-layer and dimension-independent -
which includes the HC chain, since `hc_count` is 4 in both.
