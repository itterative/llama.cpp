# E021 - the decode depth slope is a host-side scan, not KV bandwidth

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- probe: temporary `QSA_TIMING=1` fprintf around `llama_memory_hybrid_idx::set_input_qsa`
  (`src/llama-memory-hybrid-idx.cpp:273`), reverted; `llama-completion --perf` for totals

## question

The user's: after E020 made attention sparse, tg still falls off with depth (182 t/s at 40 k,
99.5 t/s at 164 k). Does the indexer still walk the whole cache, and are its block keys cached the
way KV is?

## answers

**It walks the whole cache.** `members = ggml_get_rows(k_all, inp->blk_cells)` with `blk_cells`
sized `r*n_blocks >= n_kv` names every cell, every step. It is the *indexer* cache (128 dims, one
key head, `type_k`), not the attention KV (256 x 2), so about 1/6 the bytes.

**The block keys are not cached.** `cpy_k` stores raw `index_k_proj` output; the comment at
`src/models/qwen4exp.cpp:599` says why: "cached indexer keys are raw: pooling precedes norm and
rotation". So pool -> mean -> rms_norm -> rope -> mul_mat -> relu -> head-sum -> top_k is redone
from scratch per step, which also explains `n_kv_max = 2051` (`indexer_top_k + r - 1`,
whole blocks plus the tail).

**But the decode slope is not those GPU ops - it is host code.** `set_input_qsa` fills the QSA
input tensors on the CPU, and upstream already says so:

```
// TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
//       is the per-cell scan rather than these allocations, so hoisting them buys nothing
```

Measured here, decode (one call per step, **not per layer** - the input is shared across layers by
ratio, `qsa_inps`), strictly linear:

| n_kv | us/step | ns/cell |
|---|---|---|
| 3584  | 78    | 21.8 |
| 29696 | 695   | 23.4 |

Cross-checks the upstream TODO (23 ns x 33k = 759 us vs their 865 us). Against measured step
times: 4.46 ms at depth 3.4k, 5.39 ms at 29.7k - so the scan is **13% of a decode step at 30k**
and accounts for **0.62 ms of the 0.93 ms** depth delta there, i.e. about **2/3 of the slope**.
The 40k -> 164k pair agrees: predicted scan delta 2.9 ms vs measured step delta 4.55 ms, ~64%.

Prefill pays a worse version of the same function: with `n_tps` in the thousands the bias tensor is
`O(n_blocks x n_tps)`, measured at 4-6 ms per ubatch around 25-30k depth.

## why this is a better target than anything measured so far

- **It does not scale with layer count.** Every other local cost is 1/12 of the bench's because the
  dummy has 4 layers instead of 48; a per-step host scan is identical on both boxes. So the fix is
  not attenuated by the harness, and the ~13% seen locally is a real fraction of a step, not a
  1/12 preview of one.
- **It explains an otherwise awkward null result.** E020 cut attention reads 20x and tg did not
  move; that was only surprising if one expected attention to own the slope. It does not.
- **No kernel work involved.** The mapping is append-only: a block of `r` aligned cells cannot
  change once its `r` tokens are written, so only the trailing partial block needs per-step work.
  The obstacles to check are the per-stream cell layout, `n_pad`, and context shift / rollback.

## caveats

- The probe printed per call and was averaged over 15-17 decode steps; variance within a depth was
  a few percent.
- ~1/3 of the slope is still unattributed (candidate: the GPU-side O(n_kv) mask, fill, `set_rows`,
  `get_rows` expansion and `top_k`, which sparse FA leaves fully in place).
- The ns/cell figure is this box's CPU; the bench box has a different one, so the *fraction* of a
  step it represents there will differ even though the mechanism is the same.
