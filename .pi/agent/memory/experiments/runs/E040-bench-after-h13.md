# E040 - bench after H13: tg +7.2% and pp +23% at 131k, but three changes are in that delta

`results/user/qsa-experiements/results-h13.log`, two tables.

Arm A, build `b827606c8` (11098), `GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1`:

| depth | pp512 | pp4096 | pp8192 | tg128 |
| --- | --- | --- | --- | --- |
| 4096 | 1422.19 ± 189.65 | 2184.27 ± 1.62 | 2265.16 ± 1.20 | 33.60 ± 1.34 |
| 16384 | 1413.04 ± 177.57 | 2100.95 ± 3.08 | 2156.80 ± 2.18 | 32.40 ± 1.24 |
| 40960 | 1332.45 ± 128.18 | 1919.76 ± 3.65 | 1963.90 ± 2.42 | 30.07 ± 1.03 |
| 131072 | 997.67 ± 89.94 | 1475.90 ± 5.14 | 1511.31 ± 1.25 | 23.19 ± 0.59 |

Arm B, build `84b141ac6` (11071), `Q4EXP_SPARSE_FA=1` and **no rtile**. Its `pp4096 @ d131072 = 1204.92`
is the same number E027 recorded for that build, and the table carries pasted tab characters, so this is
the earlier recorded run reused rather than a fresh run - fine as a baseline, but it is a baseline from a
different binary.

| depth | pp512 | pp4096 | pp8192 | tg128 |
| --- | --- | --- | --- | --- |
| 4096 | 1411.65 | 2137.98 | 2180.94 | 35.09 |
| 16384 | 1346.76 | 1968.42 | 2009.07 | 33.16 |
| 40960 | 1239.87 | 1720.49 | 1748.38 | 30.17 |
| 131072 | 870.23 | 1204.92 | 1227.19 | 21.63 |

A over B: pp512 +14.6%, pp4096 **+22.5%**, pp8192 **+23.1%**, tg +7.2% at 131072; at 4096 tg reads
-4.3% and at 16384 -2.3%, but with stddevs of ±1.34 and ±1.24 on ~34 t/s those are about one sigma and
are not evidence of a regression.

## Settled by `results-h13-cell-flag.log`: the prefill gain IS H13

Third table in the directory, **same build `b827606c8`**, same env, only `Q4EXP_CELL_SEL=1` added. This is
the comparison that matters, because one env variable in one binary separates the two selection paths with
nothing else in common:

| depth | pp4096 cell | pp4096 block | delta | pp8192 cell | pp8192 block | delta | tg cell | tg block | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 2182.17 | 2184.27 | +0.1% | 2204.29 | 2265.16 | +2.8% | 35.83 ± 1.45 | 33.60 ± 1.34 | **-6.2%** |
| 16384 | 2001.15 | 2100.95 | +5.0% | 2024.28 | 2156.80 | +6.5% | 34.39 ± 1.53 | 32.40 ± 1.24 | **-5.8%** |
| 40960 | 1737.45 | 1919.76 | +10.5% | 1757.58 | 1963.90 | +11.7% | 31.50 ± 1.17 | 30.07 ± 1.03 | **-4.5%** |
| 131072 | 1221.31 | 1475.90 | **+20.8%** | 1234.88 | 1511.31 | **+22.4%** | 23.39 ± 0.55 | 23.19 ± 0.59 | -0.9% |

**Prefill: H13 is the whole gain, and it grows with depth exactly as E038 predicted** for work that scales
with `n_kv x n_tps`. The cell arm at 131k reproduces the 11071 build's 1204.92 within 1.4%, so the
`n_kv_max` latching fix contributed essentially nothing to prefill - my prime suspect below was wrong, and
I should have let the one run settle it rather than reasoning about mechanisms.

**Decode: the sign is the opposite of the dev box and it is not resolved.** Same-session the block arm is
4.5-6.2% slower at 4096/16384/40960 and tied at 131072; on the dev box E039 measured +0.5%, +2.3%, +6.3%
at 8192/40960/163840. Both logs come from *different sessions* 11 minutes apart, and this bench's
cross-session floor is ~4%, which is the size of the disagreement. Against that, the deltas are consistent
in sign across three depths and each is 1.6-2.6 sigma, and the added GPU work per decode step is close to
nothing - one 2048-row int32 gather, one 2056-int concat, two view-only reshapes, minus four removed
launches and a 4x smaller top-k - which is why I do not have a mechanism for a real 5% decode loss.

Cheapest discriminator, one depth, same session, alternating: two runs per arm of
`-d 40960 -p 0 -n 256 -r 5`. If block still loses ~5% there, the regression is real and I go looking for
the mechanism instead of the noise.

## What is actually in the 11071 -> 11098 delta

`git log 84b141ac6..d77fb8d53 -- ggml src tools` shows three commits: `1d724aca0` (read `n_kv_max` per
graph build instead of latching it), `e7f0b85fd` and `d262224eb` (rtile). rtile requires `Q->ne[1] == 1`,
so it cannot move prefill. What is left for the prefill jump is H13 - confirmed above - and the latching
fix, which the cell arm now shows to be roughly neutral on prefill. The latching fix is still worth
keeping for its own reason (build-order-dependent arms in a single process), but it is not the perf story
I suspected.

`git log d77fb8d53..b827606c8 -- ggml src tools` returns exactly one commit: `e3df587e0`, H13. So the two
tg tables differ only by H13, and that comparison is valid:

| depth | 11083 (cell) | 11098 (block) | delta | sigma of the delta |
| --- | --- | --- | --- | --- |
| 4096 | 34.79 ± 1.73 | 33.60 ± 1.34 | -3.4% | 0.5 |
| 16384 | 33.23 ± 1.64 | 32.40 ± 1.24 | -2.5% | 0.4 |
| 40960 | 30.20 ± 1.23 | 30.07 ± 1.03 | -0.4% | 0.05 |
| 131072 | 22.24 ± 0.69 | 23.19 ± 0.59 | **+4.3%** | 1.8 |

(SIGMA computed from the reported per-run stddevs over 3 runs, so the standard error of each mean is
sigma/sqrt(3). The shallow dips are well inside noise; the 131072 gain is suggestive at 1.8 sigma and in
the same direction as the dev box's isolated +6.3%, but `-r 3` is not enough to call it proven.)

The prefill numbers cannot be read that way, which is why the gate run above is the one to trust: comparing
11071 with 11098 spans three commits, two of them rtile, and rtile cannot touch prefill at all.

## Still owed

- rtile engagement at shallow depth: settled by the review - the width change moves `2*n_kv_max` from
  4102 to 4104, KV rows are 256-padded, so the smallest eligible depth is 4128 before and after. No
  engagement boundary moved at any depth; nothing to confirm.
- The decode sign disagreement between machines (E039 +2.3% vs E040 -4.5% at 40960). Needs the
  same-session alternating repeat described above before I look for a mechanism.
- The selection-set differential from the H13 plan is still not done.
- Whether the whole-block rule is measurably better than the cell rule on a real checkpoint. On the dummy
  the two are indistinguishable (8e-8), which is expected for random indexer weights: agreement there
  shows nothing broke, not that the new rule is better. The argument for Eq. 19 is the report plus the
  training regime, not this PPL.
