# E055 - letting gfx12 take mmvq's small_k block shape is worth +4% to +7% of tg

The knob is `GGML_CUDA_MMVQ_RDNA4_SMALL_K` (commit in this branch, default **off**, which is upstream
behaviour). Set it to `1` to enable. E054 killed "decode is on the wrong matmul kernel"; this is the
follow-up inside mmvq.

## The mechanism, from source

RDNA4's mmvq tuning branch gives **8 warps** at `ncols_dst == 1` for exactly the types this model uses
(`calc_nwarps`, `mmvq.cu:465-489`). But two things keep the block shape at one row per block:

- `calc_rows_per_block` (`mmvq.cu:563-567`) only takes the `ncols_dst == 1 -> small_k ? nwarps : 1` branch
  for GENERIC / GCN / TURING / GB10. `MMVQ_PARAMETERS_RDNA4` is not in that list, so it returns 1.
- `should_use_small_k` (`mmvq.cu:~1090`) computes the condition and then throws it away for
  `GGML_CUDA_CC_IS_RDNA(cc)` - a blanket architecture exclusion, on the same `else if` line as the
  per-type IQ/Pascal lists, with **no comment saying why**.

The condition itself is satisfied by this model's expert shapes. For `ffn_down` at Q5_0 with
`moe_intermediate_size = 640`: `blocks_per_row_x = 640 / 32 = 20`, while
`nwarps * blocks_per_iter_1warp = 8 * (2 * 32 / 8) = 64`, so `20 < 64` -> small_k wanted. Meaning: a
256-thread block cooperatively reduces a **480-byte** row and pays a full cross-warp shared-memory
reduction per output value, with only 2.5 of 8 warps' worth of K trips to amortize it. That is the shape
`small_k` exists to fix - "increase rows_per_block to match nwarps so each warp has more work to do" -
switched off for the architecture that has the widest blocks.

Where it came from: `ec16a072f` ("Optimize MOE GEMV kernel for BS > 1.", #20905, 2026-03-29) added both
`small_k` and the RDNA exclusion in one commit. The exclusion arrived with the feature rather than after
measuring it on RDNA, which is at least consistent with "not tested there" rather than "tested and lost".

## Measurements, dev box, RX 9070, knob off vs on

`llama-bench ... -sm tensor -fa 1 -lzm on-direct -d 16384`, `Q4EXP_POOLED=0 Q4EXP_SPARSE_FA=1
GGML_FATTN_RDNA_RTILE=1`, passes interleaved. Raw: [results/E055-small-k-rdna4/](../results/E055-small-k-rdna4/).

| model / config | metric | off | on | delta |
|---|---|---|---|---|
| q4exp-4l (512 experts, 4 layers), `-sm none`, tg128 | t/s | 228.34, 227.83, 228.38 | 237.32, 237.91, 237.16 | **+4.07%** |
| q4exp-4l, `-sm tensor`, tg128 | t/s | 222.6, 222.3 | 231.4, 231.5 | **+3.96%** |
| q4exp-4l, `-sm tensor`, pp4096 | t/s | - | - | none (463.6/466.1 vs 465.1/466.6 ms) |
| q4exp-48l-12qsa, `-sm tensor`, pp8192 | t/s | 704.58, 701.66 | 704.59, 702.49 | none |
| q4exp-48l-12qsa, `-sm tensor`, tg128 | t/s | 32.60, 31.64 | 34.84, 34.15 | **+7.4%** |

Two things to take from that spread. pp does not move at all, which is expected and is a useful
consistency check: at batch 4096 > `MMVQ_MAX_BATCH_SIZE` (8) the experts are on mmq, so the knob cannot
touch prefill. And the 48-layer dummy gains nearly twice what the 4-layer does, which fits the mechanism -
this is per-expert-matmul work, so it scales with layer count, and the real model has 48.

## Correctness

- golden `-sm none` **263113.6984 +/- 3043.13362** with the knob off *and* on, i.e. bit-identical. Each
  warp still owns a whole row and walks K in the same order, so there is no reassociation - the change
  removes cross-warp reduction work, it does not reorder accumulation.
- `test-backend-ops test -b ROCm0 -o MUL_MAT` and `-o MUL_MAT_ID`: 2/2 backends passed with the knob on.

## What this does and does not claim

Claimed: on gfx1201, for these MoE shapes, one row per 256-thread block is the wrong shape and allowing
small_k is worth single-digit-to-double-digit tg with no other movement.

Not claimed:
- **RDNA3**, which the same blanket clause also excludes and which my knob deliberately does not enable
  (`GGML_CUDA_CC_IS_RDNA4` only). If RDNA3 loses, that is the reason the clause exists and the fix stays
  RDNA4-scoped.
- The `nwarps = 8` choice itself is untested against `nwarps = 1` with small_k; those two interact, and a
  real tuning pass would sweep them together.

## Confirmed on the bench box: +5.5% to +6.6% of tg, flat across depth

`results/user/llama-bench/82bc067c3/run1.log`, build `ea2c69a30` (which contains the knob - the first
attempt at this A/B ran on `82bc067c3`, before it existed, and both arms were then the same binary), 4x
R9700, real Qwen3.8-Flash-Next Q4_K_M 111.38 GiB, `-sm tensor -fa 1 -lzm on-direct -b 2048 -ub 1024 -p 0
-n 128 -r 3`, `Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1`:

| depth | knob off | knob on | delta |
|---|---|---|---|
| d4096 | 31.95 +/- 1.65 | 34.07 +/- 1.90 | **+6.6%** |
| d16384 | 32.08 +/- 1.30 | 34.09 +/- 1.47 | **+6.3%** |
| d40960 | 31.35 +/- 1.17 | 33.31 +/- 1.46 | **+6.3%** |
| d131072 | 29.19 +/- 1.13 | 30.79 +/- 1.24 | **+5.5%** |

All four depths move the same direction by 5.5-6.6%, which is far outside the ~1% run-to-run spread this
box shows at `-r 3`. The effect is flat in depth, as expected for a per-expert-matmul change and unlike the
pool, whose benefit grows with context - so this one is additive on top of H9 rather than depth-dependent.

Two consistency checks that passed. The off arm reproduces the pre-knob build within noise (31.95/29.19
here against 31.39/28.83 on `82bc067c3`), so the default-off path really is unchanged behaviour. And the on
arm lands where E050 measured 30.12 at d131072 - that overlap is across different builds, so treat it as
comforting, not as evidence.

What is still not measured here: pp on the box (this run was `-p 0`). Locally pp did not move at all, which
is the expected result since batch 4096 is above `MMVQ_MAX_BATCH_SIZE` and therefore on mmq.

## If this goes upstream

The change as it would need to look is two lines: add `MMVQ_PARAMETERS_RDNA4` to the `calc_rows_per_block`
list and drop RDNA4 from the `GGML_CUDA_CC_IS_RDNA(cc)` exclusion in `should_use_small_k`. Before anyone
proposes it: an RDNA3 data point, and a note that `should_halve_iters` declines to widen blocks for
`has_ids` ("Expert rows are gathered per token, so a wider block adds reduction work without reuse") - that
reasoning is about *more warps per row*, whereas small_k gives each warp its own row, so it does not argue
against this, but a reviewer will raise it and the answer should be ready.
</content>
