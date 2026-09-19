# E041 - reviewer-4 on H13: two defects fixed, one evidence correction, and the change is contained

Background review of `e3df587e0` (H13 block selection). The reviewer re-built, re-ran the ops gates,
re-measured all the PPL arms, and emulated the host algorithm per-query in /tmp. Verdict: the mechanism
is sound, one High defect in my tail resolution, one Medium in the ranked fallback, and several
record/evidence corrections. All code dispositions are in `93136fa41` and `9959d9508`.

## Defects

- **D1 (High, silent wrong results) - the tail had no sequence filter.** In the general path, groups are
  keyed on (position bucket, seq-set) and *prepended* (`llama-memory-hybrid-idx.cpp:406-412`), so within
  a bucket the lowest bid belongs to the most recently allocated sequence. The tail then resolved cells
  from `bid_idx` alone, so with two sequences sharing a cells array (unified KV - llama-server's default,
  `seq_cp` forks, interleaved batch) an older sequence's queries received the newer sequence's cells; the
  mask dropped them as foreign and the query silently lost its own last `r-1` tokens, up to a fully
  masked row (the NaN case the deleted "+1e9" comment used to guard). The old code was immune because the
  forward map force-included every unpooled cell. **Fixed**: the general path now builds a per-sequence
  key -> cell map in the existing per-cell pass; the tail resolves through the query's own sequence map
  and falls back to the query's own cell when a trimmed cache lost the key. The fast path keeps its
  arithmetic slot mapping and was never exposed (it requires `one_seq`).
- **D2 (Medium-High) - the fallback mixed rank space with cell space**, wrong for 3 of 4 mrope queries.
  **Fixed** by deletion: the map replaces the fallback.
- **D3 (evidence) - `Q4EXP_CELL_SEL=1` is not the pre-commit selection.** The `+1e9` removal changed the
  cell arm too, so the "old" value 267035.3653 quoted in E039/E040 is the pre-commit run; the same-binary
  cell arm measures 267035.3047. The paired comparison in E040 (same build, one env var) is still valid;
  the E039 "quality moved 8e-8" claim is corrected to "selection delta 8.3e-5 absolute, 3e-10 relative".
- **D4/D5/D6** (low): `blk_cells` pre-filled 0 can join cell 0 to the set when a row has fewer visible
  blocks than the budget; bounded to one cell and strictly better than the old arbitrary indices.
  Robustness items (can_reuse mode check, tail-to-blk_bias assert, one-block floor for a malformed
  budget, stale header doc) fixed; `width` intentionally not clamped to `n_kv`, commented.

## Corrections to prior records

- E039's relative-PPL arithmetic was off ten thousandfold: actual ratio ~3e-10, not 8e-8. Relevant
  because the dummy's near-uniform logits make the PPL gate **structurally unable to distinguish
  selection rules** - three genuinely different rules land within 8e-5 absolute.
- The rtile gate figures in E040 were wrong: `2*2051 = 4102` (not 4022), `2*2052 = 4104`, and KV rows are
  padded to multiples of 256, so the smallest eligible depth is **4128 before and after** - the width
  change moved no engagement boundary at any depth. E040's "still owed: confirm rtile at 4096/16384" is
  answered analytically, not needed.
- The `-fa 0` run in E039 did **not** exercise the `!blk_bias` fallback (mask type changes, not shape):
  the fallback is unreachable for this arch, closing E039's open item.

## Beyond qwen4exp - contained, with evidence

- `llama_memory_hybrid_idx` is instantiated for exactly one arch (`llama-model.cpp:2606,2658`) and
  `set_input_qsa` has one caller; the header is private. Sessions/state unaffected (the tensors are
  per-ubatch inputs, not serialised). All other indexer models go through `llama_kv_cache_msa`, untouched.
- The new op usages (I32 `get_rows` src0, I32 `concat`) are implemented and CI-covered on CPU, CUDA/HIP,
  Metal, Vulkan, SYCL; measured 525/525 for GET_ROWS/CONCAT/TOP_K and 7/7 FA on ROCm0.
- `test-llama-archs` / split-mode notes corrected in `qwen4exp-arch.md`: this branch re-enables
  `-sm tensor` (fork commit `a8b24dfdf`).
- Useful context: minimax-m3 already selects position blocks and expands on-device through a
  seq-filtered pos->cell map (`set_input_pos_slot`, `llama-kv-cache-msa.cpp:304-334`), so the fix
  converges on an existing in-tree pattern.

## What an upstream reviewer will ask, and the honest answer

1. Why a third block-selection mechanism? - converge or comment; the answer references minimax-m3's
   device-side expansion and why the host tail is needed here.
2. What proves the selection set? - still nothing: the promised differential remains owed, and the PPL
   gate cannot provide it. State that, don't hide behind 267035.3875.
3. Why width 2052 vs the reference 2051? - the query's own cell is always force-included so a row can
   never be fully masked; now stated in code. (This is the deletion the pre-fix comment guarded against;
   keep it deliberate.)
4. Why keep `Q4EXP_CELL_SEL` + the per-cell path + the unreachable `!blk_bias` fallback in one commit?
   - they must be split out before any submission; the gate dies after the real-model A/B.
5. Multi-sequence tail - fixed in code, but a live multi-seq run with a real checkpoint is still the
   only end-to-end proof; I could not produce it here (single GPU, blind dummy).

## Measured here (all on models/q4exp-4l.gguf, post-fix)

| run | result |
| --- | --- |
| golden corpus, new/old paths | 263113.6984 **bit-identical** (matches pre-commit) |
| deep arm, same-binary cell arm | 267035.3047 (D3's corrected "same-build old") |
| deep arm, block | 267035.3875 |
| unified 2-seq (-kvu) | runs clean, 264605.3413 |
| ops gates | 7/7 FA, 525/525 GET_ROWS/CONCAT/TOP_K |

## Still owed

The selection-set differential; a live multi-seq run; the real-model A/B through `Q4EXP_CELL_SEL`
(bench); the decode-sign repeat (E040); deleting the gate + the unreachable fallback in a later commit.