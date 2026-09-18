# E025 - bench validation of qsa-A (host fix) and qsa-B (sparse FA)

- date: 2026-09-17 | machine: bench-4x-r9700-32g (hw v1, user's fork + cherry-picks) | tier: T2 | status: open
- source: `results/user/results-qsa-experiments.log` (gitignored), `-sm tensor -fa 1 -lzm auto
  -ot per_layer_token_embd=CPU -d 4096,16384,40960,131072 -p 512,4096,8192 -n 128 -r 3 -b 2048 -ub 1024`
- arms: baseline `c9a59ef73` (11009) / qsa-A `8283df294` (11011) = fork + `3aea8533a` + `798f54bd9`
  / qsa-B `5480fa2c7` (11013) = that + `abd3473a8` + `b364ff44e`, run with `Q4EXP_SPARSE_FA=1`
- note: the log's own header for the third arm lists A's two commits twice; the commits that make
  up B are `abd3473a8` and `b364ff44e`. Recorded here so the mapping is unambiguous later.

## 1. qsa-A is confirmed, and it transfers in absolute milliseconds

| tg128 | base t/s | A t/s | base ms | A ms | saved | ns/token of depth |
|---|---|---|---|---|---|---|
| d4096   | 33.21 | 33.25 | 30.11 | 30.08 | 0.04 ms |  8.8 |
| d16384  | 31.49 | 31.62 | 31.76 | 31.63 | 0.13 ms |  8.0 |
| d40960  | 28.18 | 28.61 | 35.49 | 34.95 | 0.53 ms | 13.0 |
| d131072 | 19.79 | 20.75 | 50.53 | 48.19 | **2.34 ms** | **17.8** |

Monotone in depth, which is the signature; the individual deltas below 40k are inside their own
error bars (tg sigma is 1.5-3.6%), and only d131072 clears ~2.5 sigma on its own (+4.85%).

**Marginal rate between the two deepest points: 1.81 ms / 90112 tokens = 20.1 ns/token of depth.**
The dev box measured this scan at 23.4 ns/cell (E021), so the same code costs about 15% less time
per token there and, critically, **the saving is 1x and not 4x**. That kills the per-rank
hypothesis from E023: with `-sm tensor` across four cards the QSA input fill happens once per
step, not once per rank, and it is not multiplied by the 12 QSA layers either. E023's "this is a
reading, not a measurement" caveat is now closed by measurement: **+4.9% tg at 131k, ~0-1% below
40k, no pp change anywhere.**

## 2. qsa-B did nothing, which means it did not engage

Every qsa-B minus qsa-A delta is within +-0.9% with no structure across 12 pp rows and 4 tg rows:
pp512@d131072 +0.55%, pp4096@d40960 +0.54%, pp8192@d40960 0.00%, tg128@d131072 -0.05%.

On the dev box the identical commits gave **pp512 +23.3% at d40960 and +58.5% at d163840** (E020),
so this is not the change being weak - it is the change not running. Candidates, in the order I
would eliminate them:

- **the depth gate**: needs `K->ne[1] >= max(4096, 2*n_kv_max)`. `n_kv_max` is derived from the
  model's indexer budget, so on the real config it may differ from the 2051 measured locally;
  if it is larger the threshold moves up, though d131072 should still clear it.
- **their fork's FA dispatch**: the enablement sits in `switch_ncols2`'s RDNA branch and in
  `may_use_sparse`. If the fork already modifies `fattn.cu` (it is a performance fork), the
  cherry-pick may have landed in a path that this build never reaches, or `get_best_fattn_kernel`
  may select tile/vec instead of `mma_f16` for their shapes - note P4's finding that **decode of 1
  token uses tile/vec, not `mma_f16`**, so flat tg is expected regardless; but pp512/pp8192 should
  have used `mma_f16` and won locally.
- `max_bias` / `logit_softcap` / `mask->ne[0] == K->ne[1]` on the real file rather than the dummy.

Fix is one small probe: log inside `shall_use_sparse` and at the RDNA request, behind an env, so a
single `-d 131072 -p 512 -n 8 -r 1` run says whether the function was reached and which condition
rejected. Without it this stays uninterpretable, and B is also the instrument that decides finding
3 below.

## 3. The bigger number: 20 ms of a 50 ms decode step is depth-proportional and bytes do not explain it

From the baseline alone: 30.11 ms/token at d4096 rising to 50.53 ms at d131072, so **~20.4 ms per
token of the step is proportional to context length**, about 40% of long-context decode. E024's
inventory at 131k over 12 QSA layers puts the attention read at ~3.2 GB/token and the indexer path
at ~2 GB/token; on four R9700s (~2.3 TB/s aggregate) that is ~2.3 ms, i.e. **roughly one ninth of
the observed depth cost**. qsa-A removed 2.34 ms of it (the host scan).

So the remaining ~18 ms is per-QSA-layer GPU work that is *not* bandwidth-bound - which reframes H9
and E025: the prize is not the bytes E024 counted but the ~33 nodes per layer per step and how
badly the expensive ones run at these sizes (the `r` strided `cont()` pooling passes and the
full-cache `get_rows` are the obvious suspects). Getting qsa-B to actually engage is the cheapest
way to split "attention read" from "indexer path", because B attacks only the first.

Also worth re-reading E011 in this light: the ~28 ms fixed cost measured there was at ~40k depth,
where 0.5-2 ms of it is this host scan, and at 131k the depth term is much larger.
