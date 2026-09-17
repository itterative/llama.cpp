# E010 - graph reuse is working, and one graph build costs ~22 ms

- date: 2026-09-17 | machine: bench-4x-r9700-32g | tier: T2 | status: done | parent: E011
- build `c9a59ef73` (11009), user's fork; same model/config as E005
- variable: env `LLAMA_GRAPH_REUSE_DISABLE=1` (`src/llama-context.cpp:279`), everything else as
  E005: `-lm none -sm tensor -fa 1 -lzm auto -ot per_layer_token_embd=CPU -d 40960 -r 3`
- raw: `../results/user/results-disable-cudagraphs-reuse.csv.log` (untracked `.log`)

| test | base (E005) | reuse disabled | delta |
|---|---|---|---|
| pp512  | 397.51 +/- 8.99 | 391.54 +/- 3.34 | -1.5% |
| pp4096 | 512.04 +/- 3.59 | 513.31 +/- 3.26 | +0.2% |
| pp8192 | 546.83 +/- 6.98 | 546.60 +/- 6.95 | -0.04% |
| tg128  | 28.20 +/- 1.02 | **17.36 +/- 0.62** | **-38.4%** |

## what it says

**Graph reuse is functioning in the base config, and it saves ~22 ms per decode step**
(35.46 -> 57.60 ms/token). The CSV has no column for this env var, so the evidence that the
override applied is the 38% effect itself - which is unambiguous.

Corroborated from an independent direction: pp512 gained **19.6 ms of wall time for the entire
test** (1.2880 -> 1.3076 s), i.e. one build per test, since all 512 tokens are a single
ubatch. Two different measurements putting a graph build at ~20 ms is what makes the number
trustworthy rather than a single-row artifact.

pp8192 gained only ~6 ms despite also being one build, because in prefill the host graph work
pipelines *behind* GPU work and is partly hidden, whereas in decode the GPU is starved and any
host time is exposed in full. That is an interpretation, not a measurement, but it is the
reading that makes three different pp deltas consistent with one build cost.

**The decisive structural observation: prefill is completely insensitive and decode collapses.**
That is the per-step-fixed-cost model from E011 confirmed from a different angle - a cost paid
once per step is invisible when the step carries 512 tokens and catastrophic when it carries 1.

## what it does not say

It does **not** mean graph rebuilding is the problem. Reuse is working; disabling it makes
things worse. The ~28 ms/step of fixed cost identified in E011 exists *with* reuse enabled, so
it is separate - what E010 contributes is the **magnitude class**: host-side graph machinery of
this kind costs ~20-22 ms per pass on this box, so two or three such passes per decode step
would account for the whole unexplained gap. That converts the question from "is it host-side"
to "how many host-side passes are there", which flags cannot answer.
