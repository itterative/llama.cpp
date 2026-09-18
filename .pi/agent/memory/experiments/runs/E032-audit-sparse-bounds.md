# E032 - bounds audit of the sparse FA port after the bench faults

Background reviewer run over `abd3473a8` (ggml-cuda) and `b364ff44e` (model side), prompted by the
bench box faulting with `Q4EXP_SPARSE_FA=1` and never without it. Read-only analysis, no changes.

## Headline

**No provably unbounded index exists in the sparse path *given* the dispatch gate.** That is the
useful negative: it moves attention to three places - tensors that satisfy the gate but differ from
what the kernel then reads, the brand-new tiling instantiation, and something outside the path that
the flag merely perturbs.

## The framing fact that changes earlier confidence

The `(256, 256, 1, 16)` instantiation is **not reachable on any NVIDIA dispatch path in this tree**,
and the `256/256` sparse `test-backend-ops` cases go to `ncols2=8` on NVIDIA or to the VEC kernel at
`nb=1`. So that configuration has **zero coverage on any backend**, and the E019 ops baseline
(5633 OK) never validated it. Anything that cites the ops suite as evidence about this path is wrong;
only the `Q4EXP_FA_DEBUG` gate probe proves engagement.

Nothing in the repo exercises sparse with `gqa_ratio < ncols2` (here 12 < 16) or with
`ncolns1 < nwarps`, both of which are new to this port.

## Ruled out, with reasons

- **PDL race**: `GGML_CUDA_USE_PDL` requires `!GGML_USE_HIP`, so the launch is plain and stream-ordered.
- **`cp_async` sparse loads**: `CP_ASYNC_AVAILABLE` is off on HIP, so every sparse load is the
  synchronous 16-byte path. (Which is also why alignment is now the live concern, see R3.)
- Tail tiles / `k_VKQ_sup` / indices write coverage / padding, `mask->ne[1] >= Q->ne[1]` row bound,
  `set_rows` dst_row bound, host vs device `tile_stride` agreement, generator drift: all checked and
  bounded at this geometry. `oob_check` is forced on for sparse, so the last tile is clamped.
- **Sparse cannot engage during decode** (independent confirmation of the routing claim in E027):
  with DKQ=256 and `Q->ne[1] == 1`, `Q->ne[1] * gqa_ratio_eff = 4` fails the WMMA `> 16` test, so
  decode never reaches the MMA_F16 dispatch where sparse lives. A tg-only reproducer cannot produce
  this fault - reproduce with pp.

## Found and fixed: my `static` (1d724aca0)

`build_attn_qsa` had `static const int64_t n_kv_max = getenv(...) ? top_k->ne[0] : 0;`, so the first
graph built in the process latched the value for all later ones. Benign on the current file because
`top_k->ne[0]` is depth-independent, but with MTP draft layers carrying their own top-k width a
draft build would poison the target, and the dense-vs-sparse arms depend on build order. Now read per
build. This is an experiment-validity bug first and a correctness hazard second.

## Ranked live candidates

1. **The gather index has no upper clamp** (`indices[k_VKQ_0+i]` -> `index * stride_K`), and the kernel
   *cannot* clamp it because `ne11` has been overwritten with `n_kv_max`, so the real KV length never
   reaches the sparse kernel. Any path that leaves a scratch slot unwritten, or a pool/graph reuse
   mismatch in the row stride, converts one stray int32 into an arbitrary address. Consistent with a
   nondeterministic PC, data dependence, and a write-shaped machine lockup on a 4-card box with p2p.
   Evidence that settles it: the faulting linear address from `gpucore` / the HSA message, resolved
   against `mask->data`, the indices pool bucket, and `K->data`.
2. **The new instantiation's shared-memory budget has zero slack**: `tile_mask` starts at byte 16896,
   which is exactly the end of the K/V tile region and of the combine region; `nbytes_shared_total` is
   16976. The `j_sram >= ncols1` warp break is the only thing stopping warp 1 writing a phantom mask
   row. The dense RDNA config in use has 1280 bytes of slack, so this tightness is new.
3. **`shall_use_sparse` does not enforce the 16-byte alignment or `K->ne[1] % 256` that the gather's
   vectorised loads require** - it inherits them transitively from `gqa_opt_applies` in the kernel
   selection rule above it. Any future change to that rule turns every sparse gather into a misaligned
   access. This is the only candidate that explains "clean on one card, faults under `-sm tensor`".
   Probe extended to print `nb%16` for K, V and mask plus `K->ne[1]%256` at gate time (`tools/qsa-fa-probe.patch`);
   locally all four are 0.

## Latent, not current

The compaction kernel hardcodes `WARP_SIZE = 32` while `blockDim.x = 256`. On a wave64 device it would
silently miscount, leave scratch slots neither written nor padded, and feed the unclamped gather. Not
reachable today because `sparse_arch_ok` admits AMD only through `amd_wmma_available` (RDNA3/4, wave32)
and excludes CDNA. Worth a comment or an assert in the eventual upstream version.

## Also noted

- A row with zero selected cells yields `KQ_rowsum == 0` and writes `0/0` NaN, not a fault. If a crash
  is ever preceded by NaN in the logits, that is the mechanism, and it would be numerics not memory.
- Sparse with an empty or short top-k row is the *normal* case (causal masking) and is handled: prefix
  written, `-1` padding, loads of `-1` become zeros and `-inf` mask entries.

## Local repro attempt - negative so far

Built a dummy with the real attention geometry but 6 QSA layers instead of 1:
`models/q4exp-24l-6qsa.gguf` (`--layers 24 --experts 32 --ple-head-rows 100000 --ctx 16384`, from
`text_config` in the HF cache at `/tmp/.../allfull`), so `head_dim 256`, 24 Q / 2 KV heads (gqa 12),
`indexer_budget 2048`, `indexer_compress_ratio 4` -> `n_kv_max = 2051`, all identical to the real file.

`Q4EXP_SPARSE_FA={0,1} HIP_LAUNCH_BLOCKING=1 llama-bench -ngl 99 -lm mmap -sm none -fa 1 -lzm on
-d 16384 -p 512 -n 0 -r 1`:

- flag off: 1361.02 t/s, flag on: **1478.26 t/s (+8.6%)**, exit 0 both, no fault, no `gpucore`.

The pp gain is the engagement proof here (the probe was not applied to keep the tree clean). So at
6x the QSA-layer count, on the exact head geometry, at a depth past the gate, one card does not fault.
That leaves content-dependence (the real model's mask values) and the multi-device case as the two
surviving axes, which is exactly R1 and R3.

Note for anyone repeating this: `full_attention_interval=1` is rejected at load with
`PLE layer 1 is not a linear attention layer` - PLE only exists on recurrent layers, so raising the
QSA layer count means more layers at interval 4, not a smaller interval. Expert count is the knob
that keeps such a file inside a 16 GB card (`--experts`), and 8 layers at the real 512 experts is
13.5 GB of expert payload, which does not fit.
