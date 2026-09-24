# E054 - forcing MoE decode onto mmq is slower, so the mmq-vs-mmvq question is closed on 1 card

Ran on the dev box (1x RX 9070) against E053's finding that the 41.3% quantized-matvec group runs at
~115 GB/s, about a fifth of achievable bandwidth. The question E053 left open was whether that is a
*dispatch* mistake - `should_use_mmq` says MMQ wins on RDNA4 with `n_experts > 0`, yet decode is on mmvq.

## Method

`GGML_CUDA_FORCE_MMQ` is a compile-time define, not a runtime knob, so the test was two temporary
host-side env overrides in `ggml-cuda.cu`, both reverting one line of dispatch:

- `GGML_TEST_MMQ_MID=1`: makes `get_mmvq_mmid_max_batch` effectively 0 at the `ggml_cuda_mul_mat_id`
  call site, so `MUL_MAT_ID` falls through to `ggml_cuda_mul_mat_q` at every batch. Because the glu-fusion
  predicate consults the same cap, this arm loses the gate/up fusion as well.
- `GGML_TEST_NO_GLU=1`: clears only the glu fusion, keeping mmvq. This is the term that isolates fusion
  from kernel choice.

Build once, then interleave arms across two passes so thermal drift hits all arms equally:

```
llama-bench -m models/q4exp-4l.gguf -ngl 99 -sm none -fa 1 -lzm on-direct -d 16384 -p 0 -n 128 -r 3
  Q4EXP_POOLED=0 Q4EXP_SPARSE_FA=1 GGML_FATTN_RDNA_RTILE=1
```

`-sm none` deliberately, to measure the kernel rather than the split; the 4l dummy is the 512-expert one.

## Result

| arm | t/s (pass 1, pass 2) | vs default |
|---|---|---|
| default: mmvq + glu fusion | 228.50, 228.03 | - |
| mmvq, glu fusion off | 225.71, 225.76 | **-1.2%** |
| mmq at every batch (fusion also off) | 220.05, 219.87 | **-3.6%** |

Reverted build reproduces the default within noise: 228.26, 227.86.

So of the 3.6% gap, ~1.2% is the glu fusion and **~2.4% is mmq being slower than mmvq at `n_rows == 1`.**
The dispatch is not the problem: on RDNA4, for this shape, mmvq's tuned branch is genuinely the better
kernel, and the `should_use_mmq` `n_experts > 0` clause is a prefill statement. The ~115 GB/s has to be
explained inside mmvq itself (block shape, rows per block, the `small_k` path) or by the k-split, not by
picking a different matmul family.

## Two things worth keeping from the failure modes

- **The arm I labelled `base` was not the baseline.** It was `GGML_TEST_MMQ_MID=`, and `getenv` returns a
  non-null pointer for an *empty* value, so that arm had the override active. Its 580.8/582.3 ms numbers
  are mmq measurements wearing a base label, and the real baseline only existed in the earlier run
  because that one passed no env at all. Check `name && *name && strcmp(name, "0")` for any
  env-boolean written for an A/B before trusting a three-arm table.
- `avg_ns` in llama-bench's CSV is **per repetition**, not per token. At `-n 128` that is a 128x scale
  factor, which is how the first read of this data came out as "1.72 t/s" for a model that does 228.

## Still open

- 4-card only: whether the `ffn_down_exps` `SPLIT_AXIS_0` k-split (160 elements of k per card) pushes the
  down projection into a mmvq shape where mmq would win. This experiment was `-sm none` precisely to
  exclude that, so it says nothing about it - and it is the same one-line knob if anyone wants it on the
  bench box.
- Why mmvq reaches only ~a fifth of bandwidth here. The next thing to read is `calc_rows_per_block` /
  `small_k` rather than another A/B, since the kernel choice is now settled.
</content>
