# E057 - qsa blocks are cut in position space but sized in cell space, and an mrope draft cache splits them

- date: 2026-09-25
- machine: dev-rx9070-16g (T1)
- tier: T1
- status: done
- parent: E056 (the pooled state the field crash came from), E039 (H13 block selection), E041 (rank space for mrope)
- commit: `ebe30e1fd` (tree: clean). Pre-fix arm is `9111adf2c`, i.e. the E056 code plus the pool default flip
- build: `-O2 -g3 -fno-omit-frame-pointer`, asserts on, no `-DNDEBUG`
- model: `models/q4exp-48l-12qsa.gguf` (mechanism), `models/q4exp-4l.gguf` (gates)

## trigger

The user ran one conversation with 3 images on the bench box and got
`GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window")`
at `llama-memory-hybrid-idx.cpp:826` of that build (`:814` pre-flip numbering). The backtrace is
`results/user/issue-with-images.log`; frames 10 and 13 place it in
`LLM_GRAPH_TYPE_DECODER_MTP` under `common_speculative_impl_draft_mtp::process`, so in `ctx_dft`.

## hypothesis

The assert encodes an invariant that this arch does not hold: blocks are *indexed* by position
(`pb = pos/r`) but the block array is *sized* from the cell water mark, because
`llama_kv_cache::get_n_kv` (`llama-kv-cache.cpp:1260`) pads `used_max_p1()` to `max(n_pad, 256)`.
So the graph is in range only while `pos_max < ceil256(cell_p1)`, and a cache whose positions run
ahead of its cells violates it.

The generator is the draft context: `draft_mtp::process` returns early on embedding batches
(`common/speculative.cpp:1491-1493`), so image cells never reach `ctx_dft`, while the positions it
copies still carry the mrope advance of `max(nx, ny)` per image (`mtmd_image_tokens_get_n_pos`,
and `server_tokens::pos_next` for the accounting). The target's cells run *ahead* of its positions
(`nx*ny` cells for `max(nx, ny)` positions), so only the draft can lead.

## prediction

Written before the runs, from the arithmetic alone (`r = 4`, so `n_blocks*r == ceil256(cell_p1)`):

- the crash threshold is the *padding slack* `ceil256(cell_p1) - cell_p1` in `[0, 256)`, not depth:
  a lead of `g` fires at the first ubatch whose slack is `<= g`.
- so with `n_prompt = 300`: `g = 0` never fires, `g = 64` fires at the 3rd post-jump ubatch
  (cells 492, pos 555 >= 512), `g = 128` at the 2nd, `g >= 256` at the 1st; and moving `n_prompt` to
  100 moves the `g = 64` onset to the 2nd ubatch (window 256).
- a `g = 8` lead should survive the whole run at this phase, i.e. onset is phase-dependent, not
  monotone in depth.
- the pool is not involved: `blk_bias` comes from the mask shape and causality only
  (`qwen4exp.cpp:662`), and `qsa_pool_get` already refuses a run whose positions are not dense.
- falsified by: the crash surviving `Q4EXP_POOLED=0`, or onset tracking depth rather than the slack.

## conditions

`qsa-posgap-harness.cpp` (throwaway, in `tools/`): one plain context, `-ngl 99`, `-fa 1`,
`n_seq_max 1`, chunk 64, text only, no vision, no draft context. A run is a script of
`seqN` (N tokens at consecutive positions), `pinN` (N cells sharing one position, which is what an
mrope image writes) and `gapN` (jump the position line forward, which is what a draft cache does
when it skips image cells). It fingerprints the final next-token logits (FNV-1a over the float
bits, `n_vocab` wide), so a change in what the graph selects moves the number.

```
export LD_LIBRARY_PATH=$PWD/build/bin
g++ -O1 -g3 -Wall -std=c++17 -Iinclude -Iggml/include \
  .pi/agent/memory/experiments/tools/qsa-posgap-harness.cpp \
  -o /tmp/qsa-posgap -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin
Q4EXP_POOLED=<0|1> /tmp/qsa-posgap models/q4exp-48l-12qsa.gguf "seq300,gap64,seq1024" 64
```

## results

### the mechanism, pre-fix

| script | predicted onset | observed |
|---|---|---|
| `seq300,seq1024` | never | clean to 1324 cells, `pos_max == cell_p1 - 1` |
| `seq300,gap8,seq1024` | survives at this phase | clean to 1324 cells, `pos_max = 1331` |
| `seq300,gap64,seq1024` | 3rd post-jump ubatch | abort `:814`, last success at cells 428 / pos 491 |
| `seq300,gap128,seq1024` | 2nd post-jump ubatch | abort `:814`, last success at cells 364 / pos 491 |
| `seq300,gap256,seq1024` | 1st post-jump ubatch | abort `:814`, no ubatch completed |
| `seq100,gap64,seq1024` | 2nd post-jump ubatch (window 256) | abort `:814`, last success at cells 164 / pos 227 |

Every prediction landed, including the phase dependence, and identically in both pool arms - the
cache-creation line (`creating indexer KV cache, size = 8192 cells, block key pool = 0|1`) proves
the two arms are distinct. The window that was beaten is 512 while the cache holds 8192 cells, so
this is not a context limit.

### the change, and the two states it uncovered

`dup` (a repeated slot bit) becomes `sparse` (the position line does not step once per used cell),
tested inside `group_cells` as one `idx != prev + 1` compare, and the ranking trigger becomes
`(sparse || oor)`. Rank space is dense by construction, so `rank < n_kv` cannot exceed the window.

Feeding `pin` and `gap` together, in shapes where the duplicates exactly cancel the jump
(`seq300,pin64,gap63,seq1024`), then exposed two more states on the same path:

1. `GGML_ASSERT(n_bid <= n_blocks)` (pre-fix build, `:831`). `try_contiguous` assigns `n_bid` and
   starts writing `cur_blk_cells` *before* its per-cell verify loop, so a run whose endpoints agree
   and whose interior does not bails with a dirty block count and a half-written mapping, which the
   general path then numbers on top of. Confirmed pre-existing: the same script aborts at the same
   line on `9111adf2c`.
2. `Memory access fault by GPU`. With `pool.mode = CACHED` promised at build time from the
   endpoint-only test and the run breaking afterwards, `new_rows` / `new_cells` / `new_pos` are
   never filled - only the fast path writes them - and the pooled graph gathers indexer keys through
   them on every step. First pooled build, fresh host memory, index out of the cache.

Both are now closed: the fallback clears `n_bid` and `cur_blk_cells`, and the erase path names row 0
/ cell 0, which is in range and irrelevant because the watermark is dropped in the same branch.

### the fix, post

| script | pre-fix `9111adf2c` | A only | A + block-list clear | final |
|---|---|---|---|---|
| `seq300,seq1024` | `15cea09da05daa14` | same | same | same |
| `seq300,gap{64,256,1000},seq1024` | abort `:814` | `15cea09da05daa14` | same | same |
| `seq300,pin64,seq1024` | `5e87f5725d34cac7` | same | same | same |
| `seq300,pin64,gap{63,200},seq1024` | abort `:831` | abort `:831` | pooled=0 `5e87f5725d34cac7`, pooled=1 GPU fault | `5e87f5725d34cac7` both arms |
| `seq2000,pin512,gap511,seq5000`, 7512 cells | not run | not run | not run | `bdff01c2097ea9d6` both arms |

The equalities are the substance. Scripts that differ *only* by a position jump fingerprint like
the dense run, and scripts that differ only by the size of a jump fingerprint like the pinned run:
a hole in the position line is now inert for QSA, which is the property rank space buys and which
widening the window (the rejected alternative) would not have - it removes the abort and leaves the
cells beside each hole in a block that is never complete, i.e. invisible to attention.

## raw

The per-run logs are `.log`, so per PROTOCOL 8 they stay uncommitted (repo-wide ignore); they are
regenerateable from the command block in `conditions`, and every number quoted above is a line one
of those logs prints. The one artifact that cannot be reproduced here is the field backtrace - the
bench-box run of the 3-image conversation, real weights and 4 cards - and it sits at
`results/user/issue-with-images.log` on disk, which `results/user/.gitignore` keeps out of git on
purpose. Treat that path as a pointer to the user's machine, not as a tracked file.

Gates, `q4exp-4l`, both arms, digit-for-digit against E056's table: golden `-c 512 -b 2048`
`263113.6984 +/- 3043.13362`, sparse corpus `-c 8192 -b 2048` `267035.3875 +/- 1126.54497`. A 7.5k
cell dense run reports **0** `sched:realloc_*` in both arms, so E056's reservation result holds.

Wall, `q4exp-48l-12qsa`, 7.5k cells, pooled on: dense 16.76 s, `pin512` 16.96 s, `gap512` 15.91 s.
At or under the ~1% floor here, so no measured penalty for the ranked path - but see the next
section before believing that, because the region table and the wall clock disagree.

## open, and it is the expensive half

A pinned *or* gapped run leaves the pool (the same dense test that made the fast path bail), and the
harness then counts 21 `sched:realloc_size` at ~120 ms for `pin512`, 19 for `gap512` and 21 for the
cancelled pair, over 7.5k cells, versus **0** for the dense run at the same size (pooled=1; the
dense arm is 0 in both) - which is H19's ratchet arriving in a workload that matters. Two caveats
before it is quoted: the wall time moved by ~0.2 s, not the ~2.5 s those rows imply, so either the
counting is inflated (PROTOCOL 5: inclusive totals, unlocked counters) or the cost hides in the
device queue; and on 4 cards a re-reserve measured ~300 ms, so if the count is real this is the
dominant new cost of vision chat. Needs a bench-box reading, not more reps here.

## verdict

Accepted. The mechanism is confirmed by prediction-then-observation on a single context with no
vision and no draft, the pool is excluded in both arms, and the fix is inert on dense and mrope-
duplicate numerics while making position holes inert too.

## notes

- Three spaces exist in this path and only two of them are interchangeable: **cell** (what
  `get_n_kv` measures and what `blk_cells` names), **position** (what the block cut uses), and
  **rank** (dense over the used cells, position order preserved). The bug was a window measured in
  one space and indexed in another; the fix is that the general path now always indexes in rank.
- `qsa_pool_get` promises on the endpoint test while `set_input_qsa` verifies per cell. The lying
  state is now safe but costs a graph rebuild per 256-cell bucket. Fixing it properly means
  carrying a "verified through" cell index in the run record so both sides agree incrementally.
- `!one_seq` keeps the assert live: rank is built per cells array, so a shared cache cannot use it.
- Nothing here was found by the existing gates. The golden corpus, the sparse corpus, the greedy
  decode and the rollback harness all feed text whose positions step once per cell, which is
  exactly the state where position, cell and rank numbering coincide (and, for the golden at
  `-c 512 -b 2048`, the `n_seq = 4` trap E056 already records). None of them can produce a pin or a
  jump. `qsa-posgap-harness.cpp` is now the shape gate for this class of bug.
- The raw per-run logs are `.log` and uncommitted by PROTOCOL 8; every command above reproduces
  them, and the fingerprints are the assertion.
