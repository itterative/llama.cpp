# E044 - H9 as built: pooled indexer block keys, and the six things the design got wrong

> Code: `d8bce4e25` (`llama : pool qwen4exp indexer block keys at write time`), gate `Q4EXP_POOLED`.
> Design: [plans/h9-pooled-block-keys.md](../plans/h9-pooled-block-keys.md), prize from
> [E043](E043-review-of-e042.md). Dev box, 1x RX 9070, `q4exp-4l.gguf` unless stated. All runs
> needed `LD_LIBRARY_PATH=build/bin` - see the environment section, this bit me first.

## What landed

- `llama_kv_cache` gains an optional per-layer `pool` tensor, f32 `[idx_dim, ceil(kv_size/r),
  n_stream]`, named `cache_idx_pool_lN`, matched by the extended `pattern_idx_cache` so it is
  `SPLIT_AXIS_MIRRORED` like the raw indexer cache. Created only when the memory object asks for it.
- `llama_memory_hybrid_idx` owns the freshness state: one recorded run per ratio (`j0`, `p0`, `j1`,
  `b_lo`, `wm`), dropped by `clear`/`seq_rm`/`seq_cp`/`seq_keep`/`seq_add`/`seq_div`/`state_read`/
  `state_drop` and by `apply()` on a non-batch (update/full) context.
- `qsa_pool_get` is a pure query returning the variant: `NONE` (historic graph), `REBUILD`
  (historic chain plus a write-back of every complete row), `CACHED` (read the pool, derive only the
  blocks that completed since the watermark).
- `build_qsa_top_k`: `CACHED` reads the pool view instead of the gather/pool/norm/rope chain and emits
  a tiny in-graph chain (gather r rows, mean, scale, norm, rope, `set_rows`) for the new blocks. Both
  writing variants use existing ggml ops; no new kernel, no new file, no `tests/*` addition.

## The six corrections to the design

1. **Cell visibility timing.** `llama_context::process_ubatch` calls `mctx->apply()` *before* building
   the graph, so a block's r-th token is already committed in its own ubatch, and `cpy_k` writes its raw
   key in the same graph that then pools it. The plan's worry about a one-step lag was unfounded; the
   write belongs at completion, which is what the historic gather does. Parity depends on this.
2. **The padded window does the graph-stabilizing work.** `get_n_kv` returns
   `min(size, max(256, PAD(used_max_p1, 256)))`, so `n_blocks` - and the read view - move in steps of
   256 cells. Combined with `n_new = max(1, n_bid - wm)` (the duplicate newest row), the decode graph
   never changes shape: measured `graphs reused = 127` of 127 steps, pooled and not.
3. **Allocate on stream count, not on `unified`.** Gating on `unified` left the pool off in every
   perplexity run (they are non-unified with `n_seq_max = 1`), so the first "identical PPL" result
   proved nothing at all. The real condition is a single-stream cache: `unified || n_seq_max == 1`.
4. **An input that no node consumes has no memory.** `CACHED` deleted the only consumer of `blk_pos`,
   so the scheduler never allocated its host buffer and the old `std::fill` wrote through a NULL
   `data`: `SIGSEGV` in `__memset_avx2` at `set_input_qsa`. Inputs are allocated by reachability, not
   by `ggml_set_input`. Fix: `CACHED` does not create `blk_pos` (also kills a 512 KB/token upload),
   and `set_input_qsa` treats it as optional like `cell_blk`/`tail_cells`.
5. **The watermark has to be read live.** A reused graph carries the `pool` of the build that created
   it, so `pool.wm`/`pool.n_bid` are stale by design once the run advances. Asserting `n_bid ==
   pool.n_bid` fired at the 32nd decode token; the fill now takes `b0` from `qsa_runs[ratio].wm` and
   only uses `pool.n_new` as the width the graph can write.
6. **The REBUILD write-back was writing into the chain.** `dst` was built from `pooled`, which in that
   variant is the chain result, so the pool was never populated by a rebuild - the rows were read back
   as zeros at the following steps. Caught only because the depth sweep showed a delta exactly where
   `CACHED` reads (`-c 2048` was all-REBUILD and matched; `3072` onward did not). The tail-K probe
   pinned it: re-deriving the newest 256 rows restored bit-parity, which is only possible if the rows
   written by the rebuild itself were the bad ones.

Numbers moved by the bug hunt: `-c 8192` sparse corpus 267035.3875 (off) vs 267035.0962 (on, buggy)
vs 267035.3875 (on, fixed).

## Validation (all after the fixes, same build, env differing only)

| test | config | result |
| --- | --- | --- |
| golden corpus PPL | `-c 512 -b 2048 -np 4` | 263113.6984 both arms (matches the E041 golden) |
| sparse corpus depth sweep | `-b 2048`, `-c` 2048 / 3072 / 4096 / 6144 / 8192 | identical at every depth: 266436.6135 / 267361.5611 / 264605.3438 / 269600.9861 / 267035.3875 |
| CACHED-heavy PPL | `-b 256 -c 2048` | 261988.9423 both arms; census 3 REBUILD + 14 CACHED |
| long decode, selection non-trivial | `-c 4096 -n 2500`, greedy, seed 7 | identical 2500-token output; 2 REBUILD + 2500 CACHED |
| context-shift stress | `-c 1024 -n 2000` (repeated shifts -> seq_add/seq_div -> rebuild) | identical 16910-byte output |
| state round-trips | `test-save-load-state -c 4096` | all 8 pass in both arms (rm / add-div / keep / copy host+device / scatter / blob) |
| multi-sequence fallback | `-np 2 -c 2048 -n 400` | identical output; pool correctly off (`n_seq > 1` -> `NONE`) |

Two near-misses worth keeping: a 155-token decode test and `test-save-load-state`'s small prompts are
**vacuous** for this feature - with `n_blocks <= indexer_top_k/r` every block is selected, so wrong
pool rows cannot change anything. Any future pool test needs `n_kv > 512` (more than 128 blocks).

## Environment

- **`LD_LIBRARY_PATH=build/bin` is mandatory.** `/home/sd/.local/lib64` holds an installed
  `libllama.so.0` and the tools resolve to it first; three "identical PPL" results came out of that
  stale library and proved nothing. Check `ldd` or the `block key pool = ` log line.
- Debug recipe that keeps behavior close: `cmake -B build
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g3 -fno-omit-frame-pointer"` (same for C). It drops `-DNDEBUG`, so
  `GGML_ASSERT` fires - which is how #5 surfaced as a message instead of silent corruption - and HIP
  objects are untouched, so the rebuild stays short. The CPU path (`-ngl 0`) trips an unrelated
  pre-existing assert (`ggml-quants.c:622 nearest_int`), so CPU validation is not usable here.
- Cores: `coredumpctl --all list` (needs `--all`), `coredumpctl --all dump PID > core`, then
  `gdb -batch -ex "frame 6" -ex "print this->pool" <binary> core` - iterating on a core is much faster
  than re-running a 12.7 GB model, and `print` after `run` in the same gdb invocation loses the frame.

## Dev-box perf sanity (1x RX 9070, same build both arms)

| dummy | depth | test | pool off | pool on | delta |
| --- | --- | --- | --- | --- | --- |
| `q4exp-48l-12qsa` (12 QSA layers) | 4096 | tg128 | 35.31 +/- 0.32 | 35.08 +/- 0.38 | none, within noise |
| `q4exp-48l-12qsa` | 16384 | tg128 | 33.20 +/- 0.21 | **35.06 +/- 0.15** | **+5.6%** |
| `q4exp-48l-12qsa` | 16384 | pp512 | 903.87 +/- 27.25 | 916.72 +/- 23.48 | +1.4%, noise-level |

Flags both arms: `GGML_FATTN_RDNA_RTILE=1 Q4EXP_SPARSE_FA=1 -fa 1 -sm none -b 2048 -ub 2048 -r 3`,
only `Q4EXP_POOLED` differs. The 4096-vs-16384 contrast is the mechanism check: the chain is
`O(n_kv)`, so nothing shows while `n_kv` is small and it moves decode once `n_kv` is 16k.

Build caveat: `CMAKE_CXX_FLAGS_RELEASE` is still `-O2 -g3` with asserts live (re-running `cmake` with
only the HIP options does not reset a cached `NORMAL` string, and the bench prints
`warning: asserts enabled`). Same build on both arms, so the deltas stand, but the absolute numbers
are not comparable with the older dev-box records.

## Status and next

- Implementation: done for P1, gate default off.
- Pending: the 4-card A/B and a decode-only trace pair under the same recipe as E042, which is the
  number that matters (predicted -11 ms/token/GPU device, and prefill `O(n_kv^2)` -> `O(n_kv)`). The
  dev box cannot see the prefill effect: its chain share at 16k is small enough that pp512 is
  noise-level, so a 131k pp run is the real test of that claim.
- Then P2: drop `blk_cells`/`blk_pos` uploads (the host scan is ~3.4 ms/token at 131k, plus ~1.2 MB of
  H2D), and F8 (the FA half-count ambiguity) is still open.
- If the 4-card A/B shows the gain, flip the default to on and keep `Q4EXP_POOLED=0` for A/B.