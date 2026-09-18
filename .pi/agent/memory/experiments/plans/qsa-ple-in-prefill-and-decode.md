# QSA and the n-gram table, as they actually run in prefill and decode

Written 2026-09-18 after E031 and E027 correction 3, to answer a question the running experiments
were not answering: which parts of this model are worth optimizing at all. Anchors:

- architecture: `results/E999-qwen4exp-tech-report.md` (Qwen Team, 2026-08-26), sections 2.1.2 (QSA)
  and 2.3 (n-gram embedding). Cited below as `report 2.1.2` etc.
- code: `src/models/qwen4exp.cpp`, `src/llama-lazy-reader.cpp`, `ggml/src/ggml-cuda/fattn.cu`.
- numbers: `plans/model-shape.md` for dims, and the E-records for measurements.

Every figure is tagged: **[M]** measured on a real run, **[D]** derived from the published config,
**[A]** assumption that would change the conclusion if wrong.

## 1. The whole layer stack, and where the bytes are

```
 token ids
    |
    v
 +------------------------------------------------------------+  one forward pass, 48 layers
 |  emb 2560                                                  |
 |  +------------------------------------------------------+  |
 |  | L0  GDN + MoE      <- recurrent state, no KV          |  |
 |  | L1  GDN + MoE  *** + PLE n-gram ***                   |  |   PLE sits in exactly
 |  | L2  GDN + MoE                                         |  |   one layer (report 2.3.1)
 |  | L3  QSA  + MoE      <- 4 heads shared per KV head     |  |
 |  | ...  x12 blocks ...                                   |  |
 |  +------------------------------------------------------+  |
 |  lm_head 248320 x 2560                                     |
 +------------------------------------------------------------+

 per block of 4: 3 x GDN (linear, fixed state) + 1 x full attention (KV grows with context)
                 only the 12 attention layers can use QSA at all          [D]
 every sublayer reads and writes a 4-branch residual (GR / hyper-connection), hc_count = 4 [D]
 every MoE layer: 512 experts, 10 active, each 3 x 2560 x 640                            [D]
```

Two objects dominate the file, and neither is a GEMM:

| object | size | how it is touched |
|---|---|---|
| MoE experts | 63.3 GiB, 120.9 B params **[D]** | 10 of 512 per layer per token = 1.33 GB read per token **[D]** |
| PLE n-gram table | 32.8 GiB, 51.2 B params **[D]** | 16 rows of 110 bytes per token = 1760 B **[D]** |
| everything else | ~15 GiB | read in full, every token, ~2.1 GB **[A: per-layer sum]** |

The table is 29% of the bytes and 0.05% of the traffic per token. That single ratio is the reason
most PLE ideas are dead and the reason one of them (E031) was nevertheless worth a lot: cost here is
per *access*, not per *byte* (section 5).

## 2. What QSA does

```
                       per QSA layer, per step, over the whole cache
  cur ---- index_k_proj ---> k_raw (128/token) --cpy_k--> +---------------------------+
                                                          | indexer cache, 1 head,    |
                                                          | 128 f32 = 512 B per token |
                                                          +---------------------------+
                                                             | get_rows(blk_cells)
                                                             v                       x r=4 copies
                                            +-----------------------------+
                                            | pool: mean of every 4 cells |  -> RMSNorm -> RoPE(block pos)
                                            +-----------------------------+
                                     n_kv/4 block keys, 128 dims each
                                                             |
  cur ---- index_q_proj ---> q (4 heads x 128, token pos) ---+--> score = sum_h ReLU(q.h . k_block)
                                                                       |
                                              block-causal bias (-INF where p_b+r-1 > i)
                                                                       |
                                              ggml_top_k over n_kv cells, width 2051
                                                                       |
                                                        top_k indices -+--> attention MASK for core FA
```

- compression ratio `r = 4`, budget `K = 2048` tokens, so `K_B = 512` blocks, plus the tail of the
  current partial block: the FA mask width is `2048 + 4 - 1 = 2051` = `n_kv_max` **[M]** (probe
  printed 2051 on the bench and on the dummy).
- indexing is `O(n^2/r)` instead of `O(n^2)` **[D, report 2.1.2]**, and the *core* attention reads
  2051 of `n_kv` rows instead of `n_kv`.
- reference kernel-level effect, at 1M context: **7.6x prefill, 4.9x decode** **[D, report Fig. 6]**,
  with gains starting at 64K and the decode figure measured over 3 extra MTP steps at batch 4.

At 131k that selection ratio is 2051/131072 = 1.6%, so the arithmetic prize in the core attention is
~64x on the KV read, and the measured prize is what E027 correction 3 is: +26% to +45% on pp.

## 3. The same two steps, side by side

```
 PREFILL ubatch of 1024 tokens against 131k of cache              [D] unless tagged
 -----------------------------------------------
 indexer cache re-gathered in full, per layer   12 x 67 MB   = 0.80 GB
 pooling passes over it (4 copies, adds, norm)              ~= 1.0  GB
 score array   n_kv/4 x 4 heads x 1024 tokens f32 -> 0.54 GB, x4 head-sums
 expanded mask n_kv x 1024 tokens f32  -> 0.54 GB, and it is copied ~4 times: ~2.7 GB
 top_k over 0.54 GB
 core attention: reads 2051 of n_kv  ->  12 x 2051 x 2048 B = 0.05 GB   <- QSA wins here
                                    (dense would be 12 x 131072 x 2048 = 3.2 GB)
 PLE table: 1024 tokens x 1760 B     = 1.8 MB, spread over ~32k preads

 DECODE, one token against 131k of cache
 -----------------------------------------------
 indexer cache re-gathered in full, per layer  12 x 67 MB   = 0.80 GB   <- NOT reduced by QSA
 pooling passes                                             ~= 1.0  GB   <- NOT reduced by QSA
 score / mask arrays: 1 query only            = ~130 KB
 core attention: 2051 rows IF it could be sparse -> 50 MB
                 131072 rows because it is not   -> 3.2 GB  <- the whole gap
 GDN recurrent state, 36 layers                              = 0.11 GB
 weights (10/512 experts + dense + lm_head)                   = 3.5 GB
 PLE table: 16 rows                    = 1760 B
```

Two things fall out of this. In decode, the QSA machinery is *all overhead*: it computes a selection
worth 130 KB of arrays and saves nothing, because the kernel that runs cannot use the selection. And
the indexer's re-pool of the whole cache (0.8 to 1.0 GB per token, 12x per step) is the same order as
the entire attention read it exists to shrink - which is H9.

## 4. Why the port cannot pay in decode, and why decode is slightly worse with the flag on

Kernel routing at this shape, `head_dim 256`, `gqa_ratio 12`, `Q->ne[1]` = tokens per step:

| phase | Q->ne[1] | test in `best_fattn_kernel` | kernel family | sparse available |
|---|---|---|---|---|
| prefill | 1024 | `Q->ne[1] * 4 > 16` true | `mma_f16` **(256,256,1,16)** | **yes** |
| decode | 1 | `1 * 4 > 16` false | tile / vec | **no such code exists** |

The sparse path lives only inside `mma_f16` (`fattn.cu:238-246` dispatch, `fattn-common.cuh:1095-1102`
for the compaction and index launch, both reached from the mma launcher only) **[M by code read +
E032 audit]**. So:

- prefill gets the win **[M: +23%/+58% dev, +26..+45% bench at 131k]**;
- decode gets no win, structurally, not as a tuning matter. Getting the report's 4.9x in decode would
  mean writing a sparse variant of the vec/tile FA kernels, which does not exist in this backend for
  any model. That is a much bigger change than anything else on the backlog.

The measured decode deltas are -1.6% to -2.9%, all four depths negative, each within about one sigma,
with the dense arm running first. There is no mechanism in the graph: `n_kv_max` is an op parameter
that the vec kernel never reads, and no compaction or index kernel is launched when the sparse path
is not taken. So the candidates are (a) arm-order drift, (b) something in the fork's p2p or MTP path,
(c) my routing read is wrong. Only (a) is cheap to test: swap the order of the two arms. If it flips
sign it was order; if it stays negative the routing analysis needs re-doing before the flag moves.

## 5. The n-gram table: what a smarter cache could and could not buy

Real geometry **[D]**, from `results/user/gguf-dump.log` and `qwen4exp.cpp:1088-1131`:

```
 one tensor, 16 head-tables concatenated end to end, row = 160 weights = 110 bytes Q5_0
 head h: offset[h] 20,000,003 apart, vocab[h] ~20,000,00x rows   -> 320,001,536 rows total

 token t:  mixed_n = (t[m]*M0) ^ (t[m-1]*M1) ^ (t[m-2]*M2)        for orders n = 2..3
           row(h)  = mixed_n % vocab[h] + offset[h]               8 heads per order, 16 rows
           gathered rows -> [160 x 16] = 2560 = hidden width -> ple_key/ple_value -> gate
           -> depthwise causal conv (kernel 4, dilation 3, so 9 tokens of history per sequence)
```

Per token that is 16 reads of 110 bytes from 16 unrelated places. The useful bytes are 1760; the
*device* bytes are 16 to 32 pages (4 KiB each) = 64 to 128 KB, because a 110-byte row still drags a
whole page. That 40x amplification, not the volume, is what made this tensor hurt.

What each phase needs from it, after E031:

| | rows/s | useful B/s | page reads/s | verdict |
|---|---|---|---|---|
| decode at 36 t/s | 576 | 1.0 MB/s | ~600-1200 | one **serial** pread per row, see below; small only because these rows are cache-hot |
| prefill at 2200 t/s | 35k | 62 MB/s | ~35-70k | 32 workers in flight -> ~1 ms of wait per 100 ms |

`llama_lazy_reader::gather` picks its worker count as `min(n_readers, max(1, n/32))`
(`src/llama-lazy-reader.cpp:70`) **[M by code read]**. A decode step gathers 16 rows, so `16/32 = 0`
and it runs on **one worker: 16 preads back to back, on the critical path, inside `set_input`**. A
1024-token ubatch gathers 16,384 rows and gets the full 32. So the parallelism that made prefill fast
is structurally absent in decode, and decode is only fine because 576 rows/s over a 33 GB table on a
box with 62 GB of page cache mostly hits memory - which is also why E031 left `tg` flat. The cheap
change if that ever stops being true is a floor on the worker count (or `-lzm off`, section 5).

Options, with what each one actually buys:

| option | what it changes | expected gain | cost | verdict |
|---|---|---|---|---|
| `-lzm off` (E014, revived) | table becomes a normal CPU tensor, gather is a memcpy, no syscall or fault per row | pp: small, direct reads already removed the stall. tg: removes 16 syscalls per token, which is only worth something on a cache-miss | flag only | **run it once**, it is free, and the size now permits it: 32.8 GiB in 62.7 GiB of RAM |
| pin / `WILLNEED` the table (F1) | keeps the page cache warm instead of bypassing it | overlaps what `-lzm on-direct` already does | ~20 lines in `llama-mmap.cpp` | skip: two fixes for one problem, and ours came from upstream |
| hot/cold split of the table by n-gram frequency, hot part on GPU | turns most of the 16 reads into L2 hits | nothing measurable at these rates | converter + two-tier gather + new tensor semantics | **no**: solves a 1 MB/s problem |
| table in VRAM, split across cards (H12) | removes host reads entirely | 8.2 GiB/card would fit, so size is no longer the blocker | E007 says `-sm tensor` mirrors this tensor, so it needs real split support | **dead**, same reason as above; retract from backlog |
| store it at Q4_0 or smaller rows | 110 -> 90 B/row, fewer pages | ~20% of an already negligible cost | re-quantize 33 GB | no |

The report's own design intent is worth noting here, because it explains the shape of the thing: the
PLE layer is placed at layer 2 specifically "allowing host-memory prefetching to overlap with the
computation of the first layer" (report 2.3.1). The architecture assumes the table is *streamed with
look-ahead*. llama.cpp assumed it was resident and let the kernel fault it in, which is the failure
E031 removed. With direct per-row reads we now satisfy the assumption from the other direction, and
there is no third thing left to build here.

## 6. What this leaves, ranked by prize per unit of effort

Decode at 131k runs at 22 t/s = 45 ms/token, reading ~8-10 GB/token **[A: sum of section 3]**. That is
230 GB/s against ~2.5 TB/s of aggregate card bandwidth, so decode is *not* bandwidth-bound, and the
byte table only tells us which stalls are worth removing.

1. **The fixed per-step cost.** At 4k depth, where attention is nearly free, decode still costs
   27.7 ms/token, and E011 already measured ~28 ms/step of it as host-side. That number predates
   their p2p landing, the mmq retune and `on-direct`, so step one is a fresh baseline, not a new
   experiment: E013's `-d` sweep and E011's `-npl` sweep on the current build. If ~20 ms of it is
   still host work, it is the largest single item on the list - bigger than every attention idea -
   and E016 `perf record -g` is the measurement that would say what it is.
2. **H9, the indexer re-pool.** ~0.8-1.0 GB per token at 131k, 12x per step, in both phases, and it is
   work the reference implementation does not do per step. Byte-wise comparable to the whole KV read,
   and unlike sparse decode it needs no kernels - one cached tensor plus the invalidation problem that
   `seq_add`/`seq_div` rewrite positions. Prerequisite is still E025 (size the per-layer prize in ms).
3. **Sparse decode.** The architecture's 4.9x, and the only way to reclaim 3.1 GB/token at 131k. Needs
   a sparse variant of the vec/tile FA kernels. Not a tuning task; would be the largest change in this
   branch by a wide margin, and I would want it discussed upstream before anyone here built it.
4. **The PLE table: stop.** 1760 bytes per token. Everything in section 5 is now a rounding error, and
   E031 already banked the one real win available there.
5. **H10 (mmq for MoE shapes) is theirs**, and it is now measurable on its own merits: until E031 the
   fault stall was larger than the effect they were looking for.

## 7. Open questions this file does not settle

- Is the -2% on decode with the flag set real? Needs the two arms run in the opposite order.
- What is the composition of the ~28 ms/step at short context, on the current build? E016.
- Does the indexer really cost ~1 GB/token/step on the bench, or does tensor split shrink it? The
  node-count and byte figures come from the dev box dummy (E024) and `-sm tensor` may replicate or
  shard that work; H9's size depends on which.
- How much of the 15 GiB of file bytes I have not attributed per layer is read per token? My per-layer
  sum (~2.1 GB of dense weights) does not reconcile with file arithmetic, so one of those is wrong.
