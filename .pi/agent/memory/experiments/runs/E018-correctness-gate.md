# E018 - the dummy harness gets a model-level correctness gate

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- supersedes the artifact of E017 (same geometry, different payload fill)
- tool: `.pi/agent/memory/experiments/tools/mkq4expdummy.py` (`--fill`, `--seed`)
- corpus: `.pi/agent/memory/experiments/tools/golden-corpus.md` (frozen copy of
  `docs/development/HOWTO-add-model.md`, 12863 B - `llama-perplexity` refuses anything under
  1024 tokens, and a copy is pinned against upstream edits)

## why

E017's file is zero-filled, so every output is a function of zeros: a wrong gather row, wrong
expert index, or wrong broadcast axis returns an identical value and the run looks fine.
`test-backend-ops` compares against a CPU reference and so does cover individual kernels, but it
says nothing about how the qwen4exp graph is wired. This adds the missing model-level gate.

## the fill

Payload is pseudo-random, `default_rng([seed, crc32(tensor_name)])`, so the same args reproduce
the file byte for byte (verified: two 1.3 GB builds are `cmp`-identical, a third with a different
seed differs). Quantized tensors are filled in the byte domain rather than by quantizing random
floats, because ~10 B expert params would need a 40 GB f32 source; only the fields a block format
reads as a scale are pinned to small constants (`PIN` in the tool, offsets from `block_q*` in
`ggml/src/ggml-common.h`):

- random bytes in an f16 scale field can encode inf/NaN -> pinned to `f16(0.001)`
- a K-quant sub-block scale of **zero erases the payload** instead of shrinking it, so those get
  `0x11`/`0x01`, which yields `|w| <~ 0.03` uniformly
- floats: gammas and norms (`*_norm.weight`) sit at `1.0 +/- 0.05` so they do not squash the
  signal they scale; everything else is `N(0, 0.02)`, router logits included, so no softmax
  saturates
- `blk.N.ssm_a` is forced **negative** (uniform -2..-0.5). A near-+1 log-decay makes the GDN
  state integrate over the sequence instead of decaying; that, plus oversized quant magnitudes,
  put the first attempts at `PPL 4.3e74 +/- 3.6e74` - finite but useless, since the relative
  spread was 85%

## the gate

```
llama-perplexity -m models/q4exp-4l.gguf -ngl 99 -lm none -sm none -fa 1 \
                 -f .pi/agent/memory/experiments/tools/golden-corpus.md
```

Baseline: `PPL = 262938.7619 +/- 3039.06817`, **bit-identical across two separate runs**.
Sanity of the magnitude: `log(248320) = 12.42`, so `e^12.48` is a near-uniform distribution over
the real vocab, which is what an untrained model should give. Exact equality held here, but treat
a relative tolerance of ~1e-4 as the pass band - atomic reductions elsewhere could break bit
reproducibility without indicating a real change.

**Contract: every diff must be explained, not "diff must be zero".** The user's objection when
this was proposed stands and is the most likely future hit: porting the QSA mask compaction
(H4b) *has* to move the logits, because selecting columns and dense-with-mask agree
mathematically but not bit for bit. A diff there is expected evidence the port is live, not a
regression.

## control: does the fill change what we are measuring

| payload | tg128 @ d40960 | PPL |
|---|---|---|
| zeros (E017) | 182.40 +/- 0.55 | n/a (degenerate) |
| pattern, first attempt (magnitudes too large) | 180.30 +/- 0.61 | 4.3e74 |
| pattern, final | **181.89 +/- 1.04** | **262938.7619** |

-0.28% against the zeros build with overlapping error bars, so the fill is performance-neutral at
equal byte counts. The spread across separate invocations of the *same* file (180.3, 181.8,
181.9) says the honest noise floor on this harness is about **1%**, and that the display-driving
GPU is contributing. Rule adopted: a local delta is only interesting above 2%.

## two traps found while building it

1. **`GGML_TYPE_BF16` has a higher enum value than `Q4_0`**, so `gtype.value >= Q4_0.value` is
   not a valid "is quantized" test. The pre-existing code took that branch for BF16 and only
   worked by accident, because `GGML_QUANT_SIZES[BF16] == (1, 2)` happens to give the right byte
   count. The test is now an explicit dtype tuple.
2. **GGUF metadata arrays may not be empty**, and `ple.layers` is empty below 2 layers, since PLE
   sits at index 1. Consequence for the planned slope sweep: the ladder must keep a PLE layer and
   is `2/4/6/8`, not `1/2/4`.

## workflow rule from this run

Iterate fill or metadata changes on a ~1.3 GB smoke build (`--layers 2 --experts 32
--ple-head-rows 65536`, seconds) and write the 12.76 GB file **once**. This session wrote it
three times while hunting the logit scale, which is minutes of I/O per attempt for a change that
is invisible until the model runs.
