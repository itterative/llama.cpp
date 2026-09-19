# H13 plan - block-level QSA selection

Objective: make `build_qsa_top_k` select at block granularity and expand afterwards, as the tech report
does (Eq. 16 and 19), instead of expanding every block to its cells and running top-k over the cell
array. Motivation is measured, not guessed: E037/E038.

## What today's code does

`src/models/qwen4exp.cpp`, inside `build_qsa_top_k` (~line 644-695):

1. block scores `[n_blocks, n_idx_h, n_tps, ns]` from `mul_mat(pooled, q)`, `relu`, head-sum -> `[n_blocks, n_tps, ns]`
2. `+ inp->bias` (block-level: `-INF` invisible, `+1e9` for the tail/spare bucket)
3. `expanded = get_rows(cont(permute(score)), inp->cell_blk)` - **`n_kv` elements, plus two full copies**
4. `expanded += mask` (f32, `n_kv` wide)
5. `top_k(expanded, width)` with `width = min(n_kv, indexer_top_k + r - 1) = 2051`
6. returned as `[2051, n_tps, 1, ns]` of cell-axis indices, consumed by `build_attn_qsa` for
   `fill(-INF)` + `set_rows(zeros)` + `+ kq_mask`

Cost of steps 3-5 at 40960 on the bench, per E037: expand gather 1.16 s, the `cont(permute)` pair
1.18 s, `top_k` 3.12 s (11 dependent launches), plus one of the five f32 adds. All of it scales with
`n_kv x n_tps`, which is why E038 finds the chain's cost is 74% prefill at 131k.

## Why it cannot be done as a bit-identical rewrite

`src/llama-memory-hybrid-idx.cpp:635` force-includes the tail by scoring the tail/spare bucket at
`+1e9`, and `2048 + r - 1 = 2051` is not a multiple of `r`. So today's cell-level top-k can take three
of a block's four cells, and the tail rides along inside the spare bucket. Block-level selection is
structurally whole-blocks-only, so the selection *set* changes. That is a behaviour change and it needs
its own evidence, not an equality assertion.

## The trap that decides the host-side part

The inverse of `cell_blk` is `blk_cells`: the host writes `cur_blk_cells[(b - b_lo)*r + slot] = j`
(`:466`, general path `:580`), block-major, so block `b`'s member cells are `blk_cells[b*r + j]`. Two
problems with expanding through it:

- `blk_cells` is pre-filled with `0` (`:332`) and **the spare/dead bucket's slots are never written**,
  because the leftover cells (incomplete head *and* tail, up to `2r-2` of them) are recorded only in the
  forward map `cell_blk[j] = dead_bid`. Expanding the spare bucket would therefore gather **cell 0**, so
  the model would attend to token 0 in place of the last few tokens - silently, no crash, plausible PPL.
- the spare bucket currently carries `+1e9`, so it *would* be selected.

## Correction after re-reading the report

Eq. 19 defines the tail **per query**: `{ r*floor((i+1)/r), ..., i }` - the prefix of the query's own
block, so at most `r` tokens, and it always contains `i` itself. The port already computes exactly that
boundary for the bias (`tail_start = (q+1)/r*r`, `llama-memory-hybrid-idx.cpp:621`), on rank for mrope
and on position otherwise. Consequences:

- `tail_cells` is `I32 [r, n_tps, ns]`, written per query row, padded by repeating the query's own cell -
  legitimately a member of the set, so the padding is not a semantic change at all;
- the capacity is bounded by construction: no overflow case, no assert, no regime gate, one selection path;
- cells orphaned by an interior hole are **not** force-included. That is the single behavioural difference
  from today, where the `+1e9` spare bucket force-includes every unpooled cell - an over-broad reading of
  "always include the tail" that coincides with Eq. 19 only while positions are dense. The report defines
  no behaviour for holes, so implementing Eq. 19 literally is the defensible reading.

An earlier draft of this plan used a `[2r, ns]` list of "all cells outside a complete block", which is
neither bounded nor what the paper says, and led me to ask about gating on `try_contiguous()`. That
question is moot once Eq. 19 is taken literally.

## Changes

### Host (`src/llama-memory-hybrid-idx.{h,cpp}`)

New input tensor `tail_cells I32 [r, n_tps, ns]`, filled per query row with the block prefix of Eq. 19:
- comment why `0` is safe: `build_attn_qsa` does `set_rows(fill(-INF), zeros, idx)` and then **adds**
  `kq_mask`, so a duplicate or padding index can only ever re-write a `0` flag on a cell the mask already
  decides. A cell the mask forbids stays `-INF` no matter how many times it is unmasked. This is the
  invariant that makes the whole thing safe, so it needs to be stated where `tail_cells` is documented
  (`llama-memory-hybrid-idx.h:76-84`);
- `cell_blk` becomes unused and is deleted with its host fill: its only consumer was the expand gather
  (`src/models/qwen4exp.cpp:671`). That also removes an `O(n_kv)` int32 upload and store-fill per stream
  per ubatch, which is the store-volume cost E021/E006 measured on the host side.
- `n_kv` and the stream count were read off `cell_blk->ne[]` (`llama-memory-hybrid-idx.cpp:286-287`), so
  they have to come from the cache context instead (`get_n_kv()`, `get_n_stream()`).

### Model (`src/models/qwen4exp.cpp`)

Replace steps 3-5 with:

1. `sel = ggml_top_k(ctx0, score_blocks, K_B)` where `K_B = indexer_top_k / r` (= 512) ->
   `[K_B, n_tps, ns]` int32 block indices;
2. `cells = ggml_get_rows(ctx0, blk_cells_3d, sel2d)` where `blk_cells_3d` is `blk_cells` viewed as
   `[r, n_blocks, ns]` and `sel2d` is `sel` reshaped to `[K_B*n_tps, ns]`. Verified against the shape
   rules: `ggml_get_rows` asserts `a->ne[2] == b->ne[1]` and `a->ne[3] == b->ne[2]`
   (`ggml/src/ggml.c:3958-3959`), which is why `n_tps` has to fold into `b->ne[0]` and why the existing
   code permutes `score` the way it does. Output is `[r, K_B*n_tps, ns]` and contiguous, so
   `ggml_reshape_3d(cells, r*K_B, n_tps, ns)` is a view, no copy: within a query row the order is
   `j + r*i`, which is exactly the cell list of the i-th selected block.
3. `top_k = ggml_concat(ctx0, cells, tail_cells, 0)` -> `width = r*K_B + r = 2052`, then the
   `[width, n_tps, 1, ns]` reshape `build_attn_qsa` already expects. Concat is dispatched on element
   *size* (`ggml/src/ggml-cuda/concat.cu:224`) and `concat_cont<unsigned int, 0>` appears in the bench
   profile, so I32 is fine. `ggml_repeat` is **not** available for I32 - the CUDA REPEAT path is F32/F16
   only (`ggml-cuda.cu:5380-5382`) - which is why `tail_cells` carries the `n_tps` axis itself.

Deleted nodes: the `n_kv` expand gather, both `cont(permute)`, the f32 mask-add into the score array.
Added: one 2048-row int32 gather, one concat. Roughly flat node count, each surviving node ~4x cheaper.

Knob to keep: `n_kv_max = use_sparse_fa ? top_k->ne[0] : 0` (`:772`) is unchanged in form, but the width
moves 2051 -> 2052, which raises the rtile gate `K->ne[1] >= max(4096, 2*n_kv_max)` from 4022 to 4024 -
still under 4352, so rtile still engages at `-d 4096`. Check that against the existing
`test-backend-ops` sparse cases, which match on `n_kv_max`.

## Semantics delta, stated plainly

Selected set goes from "top 2051 cells by block score + per-cell mask" to "512 whole blocks by block
score, plus the up-to-6 leftover cells, per-cell mask applied at attention". Differences per step:
whole-block purity gained (what the paper does), up to 8 slots instead of 3 as slack, and no more partial
blocks. Expect a small PPL move in either direction; the claim to a reviewer is *closer to the
reference*, and the evidence for that is the differential below plus the paper, not a number going down.

## Validation, in order

1. **Differential on the selection itself** (the acceptance test): throwaway build that dumps `top_k` to
   CPU for the old and new code at the same state and counts differing cells per step. Expect a handful
   per layer per step, all explainable as partial-block-vs-whole-block. Without this, a PPL change is
   just a number.
2. Golden PPL, shallow: current `263113.6984 +/- 3043.13362` **will move**; record the new value, do not
   call it a regression.
3. Deep arm (`-c 8192 -f tools/sparse-corpus.md`): currently 267035.3653 dense / 267035.3524 sparse. Run
   all four combinations (old/new x dense/sparse) so the sparse-vs-dense agreement is still visible.
4. Perf: tg and pp512 at 8192 / 40960 / 163840 with `Q4EXP_SPARSE_FA` on and off, `Q4EXP_SPARSE_FA`-plus
   rtile left for a separate arm so the two effects do not mix.
5. `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 -p 'nb=1,.*n_kv_max=[1-9]'` - the rtile-eligible set
   from E035 must still pass with the new width.
6. Re-run `tools/qsa-no-indexer.patch` at 163840 to see how much of the 2.78 ms/layer chain is left.

## Expected payoff, honestly

Decode on one card: the removed terms are ~3.5-4.0 s of the 14.8 s chain at 40960 (25%), i.e. a couple of
ms/token at 131k for 12 layers - real but not transformative, and E036's point stands that the gather and
pooling (H9) are the bigger decode term.

Prefill: this is where it lives. At 131k the chain's prefill part is ~71 s and the `n_kv x n_tps`-shaped
terms are exactly the ones H13 shrinks 4x or deletes, so a 25-35% cut of prefill chain time is the
claim, against an E038 ceiling of ~+16% pp for the whole chain going away.

Not a fix for the 4-card question (H17b): if the chain is replicated into every device sub-graph, H13
helps each copy but leaves the redundancy, which is a backend-side change.

## What I need from you

Go/no-go on the semantics delta: whole blocks plus explicit leftovers, per Eq. 19, accepting that the
selected set changes. It is your QSA port, so the call is yours; if you would rather write it, the plan
above is the whole design and I will take H15 (f16 staging, no semantics at all) instead.
