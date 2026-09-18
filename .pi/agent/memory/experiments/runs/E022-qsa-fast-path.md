# E022 - contiguous-run fast path in set_input_qsa: tg +8.3% @40k, +22.0% @164k

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- code: `src/llama-memory-hybrid-idx.cpp` (`set_input_qsa`), one function
- follows E021, which identified the cost; this is the first local change that moves decode

## what landed

The per-stream QSA mapping is rebuilt from scratch every step (E021). But the usual state is
exactly one sequence whose used cells are a contiguous run of positions, and then the block mapping
is arithmetic: bucket `b = p/r`, compact id `b - b_lo`, cell of a block = `j0 + (b*r - p0)`.

So: **verify** that state in one sequential pass (no empty slot inside the run, position strictly
`p0 + (j - j0)`, run non-empty, `p1/r < n_blocks`) and, when it holds, write the outputs directly.
Everything else - holes, evictions, defrag, multiple sequences per stream, positions past the
window, the mrope duplicate/ranked case - falls through to the existing general path, which is
unchanged apart from being wrapped in `if (!try_contiguous())`. Also moved `blk_of` / `cell_grp` /
`grp_head` from value-initialized vectors to `.assign()` inside the scan, so the fast path does not
pay three `O(n_kv)` zero-fills it never reads.

Soundness of skipping the `is_pos_2d()` exclusion: `dup` can only be set when two cells collide on
a slot bit, and a run with one cell per position cannot collide, so the ranked branch is
unreachable whenever the fast path applies. The general path would have used `pos_get` too.

## numbers (dense arm, `-r 4` / `-r 2`, same binary before and after)

| metric | before | after | delta |
|---|---|---|---|
| tg128 @ d40960  | 181.80 +/- 0.78 | **196.82 +/- 0.52** | **+8.3%** |
| tg128 @ d163840 | 99.34 +/- 0.01  | **121.19 +/- 1.27** | **+22.0%** |
| pp512 @ d40960  | 6054 +/- 95     | 6043 +/- 92    | noise |
| pp512 @ d163840 | 2532.6 +/- 3.8  | 2548.8 +/- 29.3 | noise |
| tg128 @ d40960 + `Q4EXP_SPARSE_FA` | 181.86 +/- 0.92 | 197.21 +/- 1.02 | +8.4%, composes with E020 |

The gain growing with depth is the signature of removing an `O(n_kv)` per-step term, and pp is
untouched, which is right: prefill is GPU-bound and this is host code.

## correctness

Four arms, all bit-identical to the pre-patch values from E020/E021:

| arm | PPL |
|---|---|
| golden corpus, `-np 1`        | 262938.7619 +/- 3039.06817 |
| sparse corpus, `-c 8192` sparse | 267157.2589 +/- 1127.37697 |
| sparse corpus, `-c 8192` dense   | 267157.4202 +/- 1127.37959 |
| sparse corpus, `-np 2 -c 4096`  | 264558.5414 +/- 1031.43112 (see caveat) |

**Gate caveat found on the way:** the `-np 2` arm is **not reproducible run to run** on the *unmodified*
code - 264558.5345 / .5356 / .5109 / .5306 across four runs, ~1e-7 relative, since parallel sequences
share the cache and slot ordering is timing dependent. So the E018 fingerprint is bit-stable for
`-np 1` only; the 1e-4 relative band recorded there absorbs this, but "exact match" is not a
statement that holds for multi-sequence runs.

## the mistake worth remembering

The first attempt bailed on **every** call because I required `!ubatch->is_pos_2d()`, assuming text
runs are not 2D-positioned. qwen4exp uses mrope-style sections (`[11,11,10,0]`), so `is_pos_2d()` is
true for plain text: `fast=0 bail pre=99`. tg read 182.25 (+0.25%, noise) and looked like "the scan
wasn't the cost after all" - it was the precondition. Only counters proved it. Lesson: a null result
from an optimization that never executed is indistinguishable from a wrong hypothesis; measure that
the new code ran before believing what it did not do.

## what is left on the table

The fast path still walks all cells twice - verify, then write. Saving 1.87 ms of the ~3.8 ms scan
predicted at 164 k leaves ~12 ns/cell. Two ways down further:

- **one pass**: write every bucket as full while scanning, then patch the <= r-1 trailing cells of
  the last incomplete bucket to the spare block. No new state, roughly halves again.
- **memoize per stream**: the mapping is append-only once a block is complete, so a cached copy plus
  an incremental extend would make it O(r) per step. Needs a cheap, sound mutation signal from the
  cells (evictions, defrag and context shift must invalidate), i.e. new state in the memory object -
  a bigger change, worth doing only if the one-pass version is not enough.

And ~1/3 of the original depth slope is still GPU-side O(n_kv) work (mask fill, `set_rows`, the
`get_rows` expansion, `top_k`), which this change does not touch.
