# E047 - rtile vs the generic dispatch across query-tile counts: nb=7 costs 1.5-2.1x nb=1, and vec is never picked

Asked by the batch-scaling question after `d2319c937`: how much does a multi-token pass actually cost
against a single-token one, and what runs when `GGML_FATTN_RDNA_RTILE` is unset. T1, op level, one op
at the qwen4exp QSA shape (D=256, 2 KV heads, gqa 12, f16 KV, `n_kv_max=2048`). Raw:
[results/E047-rtile-batch-sweep.txt](../results/E047-rtile-batch-sweep.txt).

**Hypothesis written first:** one block per query means cost grows roughly linearly in nb once the GPU
is full, so nb=7 should be far worse than nb=1 for rtile and much better for a kernel that packs
queries; expected effect: rtile/nb=7 within 1.5x of rtile/nb=1 x 7 if traffic-bound, near 1x if the
device was idle at nb=1.

## The numbers (us/run)

| kv | nb | rtile off | kernel | rtile on | rtile/off |
| --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 32.56 | tile | **29.84** | 1.09 |
| 16384 | 1 | 98.56 | tile | **38.54** | 2.56 |
| 32768 | 1 | 189.8 | tile | **51.1** | 3.71 |
| 32768 | 2 | 438.24 | tile | **67.70** | 6.47 |
| 32768 | 4 | 442.66 | tile | **77.53** | 5.71 |
| 32768 | 7 | 160.19 | mma_f16 | **105.75** | 1.51 |
| 32768 | 8 | 177.40 | mma_f16 | **147.50** | 1.20 |
| 32768 | 16 | 308.95 | mma_f16 | **217.02** | 1.42 |
| 131072 | 1 | 778.20 | tile | **125.34** | 6.21 |
| 131072 | 2 | 2011.22 | tile | **144.53** | 13.9 |
| 131072 | 4 | 1754.95 | tile | **152.45** | 11.5 |
| 131072 | 7 | 241.60 | mma_f16 | **182.65** | 1.32 |
| 131072 | 8 | 260.97 | mma_f16 | **224.64** | 1.16 |
| 131072 | 16 | 435.90 | mma_f16 | **307.48** | 1.42 |

## Answers

**nb=7 is not much slower than nb=1.** Inside rtile: 51.1 -> 105.75 us (2.07x) at 32768 and 125.3 ->
182.7 us (**1.46x**) at 131072. Per token that is 15.1 us and 26.1 us against 51.1 and 125.3, so a
7-wide verify pass is 3.4x (32k) to 4.8x (131k) cheaper *per token* than walking 7 single tokens. The
sub-linear part is occupancy, not reuse: `ncols1 = 1` means the grid goes from 6 to 42 blocks and the
device was mostly idle at nb=1. Which also bounds the claim - the traffic each query does is not
shared, so this cannot keep improving past the point where the SMs are full.

**Nothing falls back to vec.** `gqa_opt_applies` is true for this shape (gqa 12, mask present, no
alibi, `K->ne[1] % 32 == 0` - [fattn.cu:550](ggml/src/ggml-cuda/fattn.cu#L550)), and the unquantized
vec branch additionally requires `!gqa_opt_applies && nb == 1`
([fattn.cu:695-706](ggml/src/ggml-cuda/fattn.cu#L695)), so vec is never eligible here. The baseline is
**tile** at nb=1/2/4 and **mma_f16** at nb=7/8/16, which is what the new `GGML_FATTN_DEBUG=1` line
reports per case, and it explains the non-monotonic baseline: `nb * gqa_ratio_eff > 16` is the WMMA
threshold ([fattn.cu:690-692](ggml/src/ggml-cuda/fattn.cu#L690), `gqa_ratio_eff = 4` at D=256/gqa 12),
so nb=4 stays on tile while nb=7 crosses into mma_f16 and gets *cheaper in absolute terms*
(442.66 -> 160.19 us).

**So the commit-message ratios were mixing two comparisons.** rtile/off at nb=1/2/4 is rtile against a
kernel that ignores `n_kv_max` entirely (only three `mma_f16` shapes have a sparse instance, and
D=256 with `ncols2=4` is not one of them, so tile walks the whole cache). At nb>=7 the baseline is
`mma_f16`, also without the gather but with query packing, and that is the honest MTP comparison:
**1.16-1.51x on a verify pass**. The 6-14x figures at nb=2/4 measure "gather vs walk", not "our kernel
vs a good fallback".

## Not shown by this

Wall clock. E046 covers that: this op-level gain is ~1.2% of a decode step at 12 QSA layers, and the
single reported MTP number could not resolve it. Op-level wins here, end-to-end attribution there.

Status: done (T1). Side effect kept in the tree: `GGML_FATTN_DEBUG=1` logs the chosen kernel per
distinct (kernel, shape), which is the thing that made this answerable at all.