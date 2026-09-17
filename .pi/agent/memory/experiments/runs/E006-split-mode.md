# E006 - tensor split vs layer split on the bench box

- date: 2026-09-17
- machine: bench-4x-r9700-32g
- tier: T2
- status: done
- parent: E005
- build: `c9a59ef73` (11009) - user's fork, **same build as E005**
- model: `bartowski/Qwen3.8-Flash-Next-GGUF` Q4_K_M, ~6 B active (see E005 F2 note)
- variable: **`-sm tensor` -> `-sm layer`, nothing else** in the command line
- held constant: `-lm none -fa 1 -lzm auto -ot per_layer_token_embd=CPU -d 40960 -p 512,4096,8192 -n 128 -r 3`

## results

| test | tensor t/s | layer t/s | layer/tensor |
|---|---|---|---|
| pp512  | 397.5 +/- 9.0  | 347.5 +/- 19.3 | 0.87 |
| pp4096 | 512.0 +/- 3.6  | 359.9 +/- 1.7  | 0.70 |
| pp8192 | 546.8 +/- 7.0  | 369.0 +/- 4.1  | 0.68 |
| tg128  | 28.20 +/- 1.02 | 24.73 +/- 0.28 | **0.88** |

Raw: `../results/user/results-layer.csv.log` (untracked `.log`, per `PROTOCOL.md` 8); the
tensor column is `../results/user/results.csv.log` from E005.

Method caveat: this is **not** an interleaved A/B/A/B. The two runs are ~33 min apart
(14:35-14:38Z and 15:12Z), so thermal and clock drift are uncontrolled. It is still the
first one-variable A/B on this branch, the deltas (12-32%) exceed every reported spread,
and the conclusions below are *sign* conclusions, which survive drift.

## What it refutes

**My E005 mechanism (cross-GPU collectives) is dead, on sign not magnitude.** With
`-sm tensor` every split matmul needs a cross-GPU reduce - roughly 6 matmuls x 48 layers =
~288 collectives per token - and with `-sm layer` there are essentially **none**. So if
collectives dominated, layer split would be the *faster* decode mode. It is slower. I had
written that 288 x 50-150 us "brackets" 35.5 ms/token; the arithmetic happened to land in
range, but the direction was wrong, and a mechanism that predicts the wrong inequality is
refuted regardless of how well its numbers fit. The latency estimate was also too generous
to PCIe for a custom P2P AllReduce (a few us to low tens of us, not 150).

**Bandwidth-bound is also dead.** 6 B active at Q4_K_M is ~3.3 GB of weights per token.
Under layer split the cards are used **serially** - one active at a time - so the tg floor is
`3.3 GB / one card's bandwidth`, roughly 4x the tensor-split floor where all four read in
parallel. Predicted layer/tensor ratio if bandwidth-bound: **~0.25**. Measured: **0.88**.
A 12% penalty where 75% was predicted.

**And L1/F1's premise is dented.** Page-faulting the PLE table is *identical* in both modes,
so it cannot explain the 12% difference - but more importantly it was always a candidate for
the absolute tg cost, and the absolute cost is the same in both modes. That is consistent
with it, not evidence for it. F1 stays open; it is just no longer the leading explanation by
default.

## What it establishes

1. **`-sm tensor` wins tg here, which matches the user's expectation** (their experience: for
   decode, tensor always wins). My earlier suggestion that layer split might be preferable for
   decode is dead.
2. **pp under layer split did something it should not have done.** Corrected per the user:
   *usually* layer split wins pp, because layers run in a pipeline across cards so the
   inter-card transfers are hidden behind compute overlap between layers (and may be smaller
   in volume than tensor's). Here layer lost pp by 12-32% and barely scaled with depth
   (x1.06 from 512 to 8192 vs tensor's x1.38). So this is **not** confirmation that "pp wants
   parallel cards" - that assertion was mine and it rested on a false baseline. It is an
   anomaly needing its own explanation.
3. The anomaly and the tg insensitivity have one candidate that explains both: the **CPU split
   at layer index 1** from `-ot per_layer_token_embd=CPU`. A host stall mid-graph destroys the
   very inter-layer overlap that makes layer-split pp fast, *and* it is mode-independent, so it
   hits tg in both. That makes this pp result supporting evidence for F3 rather than a separate
   finding - the opposite of what I claimed from the same numbers an hour earlier.
4. Still true regardless: the tg cost is serial and shared - split mode moves it 12% where
   bandwidth predicted 75% and my collectives theory predicted the wrong sign.

## Leading hypothesis now

Something per-token that is host-side and mode-independent. Two ranked candidates, both
flag-testable:

- **(a) the `-ot per_layer_token_embd=CPU` placement.** It forces a CPU gather + H2D copy +
  graph split at layer index 1 on every step, and the n-gram hash is host-side anyway. It may
  additionally be what stops HIP graph capture for the whole graph - if the graph cannot be
  captured, decode degenerates into eager per-node submission from one host thread, which at
  a few thousand nodes per token is comfortably tens of milliseconds and would look exactly
  like this. That chain also explains the 9% utilisation: the GPUs are waiting to be told
  what to do.
- **(b) plain launch/sync overhead per token** (no graph capture for other reasons, sampler
  readback, KV bookkeeping).

Note (a) and (b) are not alternatives so much as: (a) is the most likely *cause* of (b).

## Next: E007 and E008, both flag-only

| id | change | if the hypothesis is right |
|---|---|---|
| ~~E007~~ | ~~drop `-ot per_layer_token_embd=CPU` and put the table on GPU~~ **killed, and I was wrong to propose it** | under `-sm tensor` the PLE table is **mirrored, not split** (`src/llama-model.cpp:513-515`), so ~30 GB on each of 4 cards is ~120 GB of the 128 GB box - my "~48 GB free so it fits" arithmetic ignored the replication entirely. Replaced by `plans/ple-prefetch.md`, whose leading option is a few-hundred-MB VRAM **row** cache rather than the whole table |
| E008 | keep today's config, add `GGML_CUDA_DISABLE_GRAPHS=1` | tg barely changes - which would show graphs were **already** not being captured, corroborating (a) independently of E007. If tg instead gets much worse, capture *is* active and the cost is elsewhere |

E008 is the more informative of the two per minute spent, and it cannot make anything
worse permanently since it is one env var.

Caveat to record honestly: `prefetch`/`-lm` was **not** varied as E005 suggested - the user
changed only `-sm`. That is the better experiment (one variable), and it is why the
refutations above are clean; but it leaves the mmap-vs-none question untouched.
