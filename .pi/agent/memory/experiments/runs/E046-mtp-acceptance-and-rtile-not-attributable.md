# E046 - MTP at 0.25 acceptance with 6 drafts: 22.77 t/s, and why the rtile change is not attributable

**Hypothesis (written first):** a verify pass is 7 queries over 12 QSA layers, so letting rtile cover
`nb > 1` (`d2319c937`, op-level numbers in `runs/E045...` follow-up and the commit message) should
shorten it. Expected size, from the measured op times at
D=256/gqa 12/budget 2048: `12 x (242.3 - 184.0) us = 0.70 ms` per step at 131k-class depth, on a step
of ~59 ms -> **+1.2%** (up to ~1.7% if the depth is their 245760). Deciding metric: `tg` with
`draft-mtp` at the same acceptance rate, `GGML_FATTN_RDNA_RTILE=0/1`, interleaved, >= 3 repeats.

## What arrived

One number, from a different session than the comparison arms: **22.77 t/s at acceptance 0.25 with 6
draft tokens**. Prior context, other sessions: no-spec **34.86**, mtp after the clamp **23.06**
(acceptance 0.26), mtp before the clamp **21.5** (0.23). Per non-negotiable 2 that is
**unmechanised and not attributable**: no same-session A/B, no repeat spread, and acceptance moves
between every pair.

For what it is worth as a sanity check rather than evidence: acceptance alone predicts most of the
movement. Tokens per step are `1 + sum(alpha^k, k=1..6)`, i.e. 1.3513 at 0.26 and 1.3333 at 0.25, so
23.06 at 0.26 normalizes to 22.75 at 0.25 - within 0.02 t/s of what came back. That is consistent
with the expected +1.2% being invisible at this resolution, and equally consistent with it being
absorbed by dispatch gaps (E039/E042 already measured this box as gap-dominated). It does not
support either conclusion.

## The arithmetic that is session-independent

At acceptance 0.25, speculative decoding is capped at **+33% tokens per step even if the 6 draft
replays were free**. Observed: MTP is 22.77 against no-spec 34.86, i.e. 54% more tokens/step at 53%
*lower* throughput. Per step: `1.3333 / 22.77 = 58.6 ms` versus a no-spec token at 28.7 ms, so 6 draft
forwards plus a 7-wide verify cost **~30 ms**. FA is ~2 ms of that. Whatever is wrong with MTP here is
not in a kernel, which is what H18 already said and this re-confirms from the other direction.

## Why acceptance might be 0.25 at all (new, cheap to check)

Reading `common_speculative_impl_draft_mtp` (`common/speculative.cpp:1330-1432`):

- `n_mtp_layers = max(1, llama_model_n_layer_nextn(model))` - a GGUF with **zero** nextn layers still
  reports 1.
- `chain_heads = n_mtp_layers > 1 && !is_mem_shared`, and `params.n_max = min(n_max, n_mtp_layers)`
  happens **only under `chain_heads`**. So a single-head (or head-less) file is not clamped: it drafts
  all 6 steps by looping the one head.
- the head tensors are optional at load (`src/llama-model.cpp:1639-1704`, `TENSOR_NOT_REQUIRED` on the
  `blk.%d.nextn.*` family), so a file without them loads fine and drafts from whatever is there.

`plans/model-shape.md` records `supports_mtp_export = False` / `no_mtp` for the converter we read, which
means the 0.25 is not necessarily a property of the model - it may be an absent draft head. Check on
the bench box, free: `qwen4exp.nextn_predict_layers` in the GGUF metadata, `grep -c 'blk\..*\.nextn\.'`
over the tensor list in the load log, or `strings <file>.gguf | grep nextn`. If those tensors are
missing, MTP is off the table for this checkpoint and the whole H18 thread closes as "not a bug".

## Next, in order of information per minute

1. The `nextn` presence check above (grep, no GPU time).
2. Same-session `RTILE=0/1` A/B with acceptance reported per arm, 3 repeats - the only way to make the
   +1.2% claim or drop it.
3. `--n-draft 1,2,4,8` (H18's diagnostic): the slope of ms/token vs draft count is the per-replay fixed
   cost. At acceptance 0.25 the expected gain from draft 2..6 is 0.021 tokens/step, so if the slope is
   anything like linear, small `n_draft` wins and MTP should probably just be off.

Raw: none this record (single user-reported number, logged as such).