# E039 - H13 block-level selection: +6.3% tg, +6.8% pp at 164k, quality-neutral

Implements [plans/h13-block-selection.md](../plans/h13-block-selection.md) on the dev box (RX 9070,
dummy `q4exp-4l.gguf`, one QSA layer). Both arms are the same binary, separated by
`Q4EXP_CELL_SEL=1`, with `Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1` in both, `-r 2`:

| depth | pp512 old | pp512 block | delta | tg128 old | tg128 block | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 7543.11 | 7529.30 | -0.2% | 233.93 | 235.00 | +0.5% |
| 40960 | 6476.89 | 6603.54 | **+2.0%** | 206.67 | 211.36 | **+2.3%** |
| 163840 | 3744.86 | 4001.02 | **+6.8%** | 138.55 | 147.31 | **+6.3%** |

The `old` column is a valid baseline: 233.93 / 206.67 / 138.55 against E036's recorded
237.86 / 207.54 / 140.44 for the same arm on this box (1.7% / 0.4% / 1.3% off, within the dev floor)
and rtile is engaged in both, since these track E035's rtile numbers.

Quality, deep arm (`-c 8192 -f tools/sparse-corpus.md`), which is the shortest context where the 2048
budget actually bites on this dummy:

| arm | old selection | block selection |
| --- | --- | --- |
| dense FA | 267035.3653 | 267035.3875 |
| sparse FA | 267035.3524 | 267035.3629 |

+8.3e-8 and +3.9e-8 relative, and the dense-vs-sparse agreement is preserved. The shallow golden corpus
(`tools/golden-corpus.md`, 512 cells) is **bit-identical** at 263113.6984 in both arms - not evidence of
nothing, but the expected invariant: with `n_kv` under the budget every block is selected, so the two
paths must agree exactly, and they do.

## The wrong measurement I made first, and why it matters

The first A/B read +86% pp and +46% tg at 164k. It was bogus: my gate conflated two flags and routed
"old" to the *fallback* path (no per-block bias, per-cell bias array uploaded, expand + top-k over
n_kv), which is far slower than the old default at depth. The tell was that the "old" arm came out at
101 t/s against E036's recorded 140 for the same configuration - **comparing against a baseline that
disagrees with the last measurement of it is the check that catches this**, and it is worth doing before
quoting any number. Fixed by splitting the flags: `blk_bias` decides the bias layout, `block_sel` decides
the selection, and the host tells them apart by which index tensor it was given.

## What the change actually deletes

- model: the `n_kv` expand gather, both `cont(permute)` copies, the f32 per-cell mask add, and top-k over
  `n_kv` -> top-k over `n_blocks` plus one int32 gather of 2048 rows and one concat;
- host: `cell_blk`, the I32 [n_kv, ns] forward map - its only consumer was that expand - so the per-step
  O(n_kv) store volume E021/E023 measured as the residual host cost is halved at the same time.

Selection is now 512 whole blocks plus Eq. 19's tail, width 2051 -> 2052. Per the plan, this is a real
behaviour change: cell-level selection could keep 3 of a block's 4 cells, and cells orphaned by an
interior hole were force-included by the `+1e9` spare bucket and now are not.

## Still open

- The +6.3% is the dummy with **one** QSA layer and the host cost counted once per step, so it does not
  transfer directly: on the real model the GPU terms are x12 while the `cell_blk` fill stays per-step.
  Needs the same A/B on the bench box at `-d 131072`.
- Coverage gap: the `!blk_bias` fallback (per-cell bias, no block selection) is not exercised by any
  measurement here. `-fa 0` runs clean and gives 263113.5607, but I could not confirm which branch it
  took, and it may be unreachable for this arch in practice. If it is dead, a reviewer will ask why it is
  kept, so that is worth settling before any submission.
- The rtile gate threshold moves with the width: `K->ne[1] >= max(4096, 2*n_kv_max)` is now 4104 instead
  of 4022, so rtile stops engaging below about 4104 cells per device. At `-d 4096` the bench showed 4352,
  which still clears it, but the margin is now 248 cells rather than 330.
- The `Q4EXP_CELL_SEL` gate is temporary and must be deleted, along with the whole per-cell path it
  protects, once the bench A/B is done (the fallback for `!blk_bias` stays - it is a different thing).
- The promised selection-set differential (count differing cells per step) was not done; the PPL delta is
  standing in for it, which is weaker evidence for a reviewer.
