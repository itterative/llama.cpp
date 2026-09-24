# E048 - `--n-draft 6` was the wrong knob: 3 drafts is much better than 6 on the 4-card box

Informal user run on the bench box (server, `-sm tensor`, real checkpoint, `draft-mtp`, ctx in the
245760 range). **No raw output and no acceptance rate reported**, so this is a data point, not a
measurement - the numbers below are as quoted in chat.

- `--n-draft 6`, acceptance ~0.25: **22.77 t/s** (E046).
- `--n-draft 3`: **37-57 t/s depending on what is being done**.

The spread in the second number is why this can't be read as a ratio: the range is wide enough that it
covers both "modest gain" and "roughly doubled", and the no-spec reference for the same session is
unknown (E045-era reference: 34.86 t/s). What is solid: cutting draft count from 6 to 3 is worth tens of
percent, and no kernel work was involved.

## Why this was predictable from the model, not the machine

Tokens per step with per-position acceptance `a` and `n` drafts is `1 + sum(a^k, k=1..n)`. At `a = 0.25`:

| n | tokens/step | share of the n=6 yield |
| --- | --- | --- |
| 1 | 1.2500 | 93.8% |
| 2 | 1.3125 | 98.4% |
| 3 | 1.3281 | 99.6% |
| 6 | 1.3333 | 100% |

So drafts 4-6 buy **0.4%** more accepted tokens per step and cost 3 more draft decodes plus a wider
verify pass. At the ~4.9 ms/draft-step implied by 22.77 t/s against a 28.7 ms no-spec token (E046's
arithmetic), removing them is worth ~15 ms/step. The same table also predicts n=1 as the optimum at
this acceptance - worth a run, since the yield difference between 1 and 3 is 6% while the cost
difference is 2 replays.

Caveat that matters for reading this: `a` is not constant. Fewer drafts usually means a higher
per-position acceptance, so the table under-predicts small `n`. That is the most likely reason the
measured gain beat the prediction.

## Follow-up

1. Re-run n = 1, 2, 3, 6 in one session with `drafts`/`accepted` counts per arm, so the yield is
   measured instead of assumed, plus the no-spec reference in the same session.
2. Then the accounting question: with `n = 3` the remaining gap is small enough that H18's "is it the
   draft ctx's host cost" may stop mattering. The split is available from existing llama.cpp perf data
   (see the H18 note on `common_perf_print` and the draft ctx being invisible to it).
3. `--n-draft 3` (or lower) should become the recorded default for this box until acceptance is
   measured properly - H18's headline "MTP costs a third of decode" is a property of n=6, not of MTP.