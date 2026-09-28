# E076 - the lazy PLE gather at prefill width: is the WILLNEED loop worth its syscalls, and is the gather on the critical path

- date: 2026-09-28
- machine: bench-4x-r9700-32g (4x gfx1201)
- tier: T2
- status: **superseded, never run** - E058 had already closed the knobs these arms test (row caches,
  `POSIX_FADV_RANDOM`, the worker divisor, `-lzm off` as a residency test), and the user's call was to
  implement the prefetch instead rather than re-measure them. See **E077** for the pre-registered
  predictions and the dev-box result of that implementation. The composition analysis below is kept
  because it is what named the target.
- parent: E074 (the same runs' region tables), E058 (`LLAMA_LAZY_PREFETCH` and `LLAMA_LAZY_WORKERS`, tuned at decode
  width), E065 (prefill phase inventory on the dev box).
- raw: [results/user/E076-lazy-prefetch-at-prefill-width/](../results/user/E076-lazy-prefetch-at-prefill-width/) (gitignored)

## question and what decides it

From E074's `prof-run.log`, per 1024-token prefill ubatch (56 calls, all of these nested inside `graph:set_inputs`):

| nested region | ms/call | what it does |
| --- | --- | --- |
| `graph:set_inputs` | 89.2 | staging step of a prefill call |
| `input:lazy_gather` | 79.2 | the PLE row gather |
| `lazy:prefetch` | 70.8 | `posix_fadvise(POSIX_FADV_WILLNEED)`, one call per distinct row |
| reads, `to_float`, sort | ~7.8 | the row reads and the q -> F32 conversion |

Counters, same run: `io:rows_prefill` 16.38k/call (16 per token, `ple_n_heads`), `io:uniq_prefill` 16.38k/call
(no repetition inside a call), `io:rchar_prefill` 1.80 MB/call (so a row is ~110 B), `io:storage_prefill`
36.75 MB/call (a 4 KB page fetched per ~110 B row, some pages shared). So the storage traffic is 20x the
requested bytes and the whole gather is 89% fadvise syscalls, one per distinct row.

Two questions, both settled by the same table:

1. does the per-row WILLNEED loop still pay at prefill width (16384 rows), or is it a syscall storm whose
   work belongs in bulk elsewhere?
2. is the gather on the critical path? E074's phases sum to roughly the wall, per 1024-token ubatch:
   device ~279 ms (`graph:compute`), host ~157 ms (`set_inputs` 89, `alloc` 20, `build` 1), `phase:sync` ~49,
   so ~79 ms of gather is either serialized or hidden behind the previous ubatch's drain.

Deciding metric: `input:lazy_gather` and `lazy:prefetch` ms/call (primary), `io:storage_prefill` and
`io:rchar_prefill` bytes/call (to prove *which* reads happen), `pp4096` t/s at `-d 16384` against E074's
2138.37 (fusion on) and 2125.42 (reference table). Never a traced number; `GGML_PROF_REGIONS` is host-side only.

Motivation for the arms is a design the user proposed: prefill knows the next batch's n-gram rows host-side,
so the expensive part can be issued during the current batch's compute window. That only pays if the gather is
serialized (question 2) and if the current per-row prefetch is the wrong instrument at this width (question 1).
The arms below price both before any code is written.

## arms

All arms: the E074 command line, `-p 4096 -n 0 -d 16384 -r 5 -b 2048 -ub 1024`, `GGML_PROF_REGIONS=1`.
Interleaved as `A B A B`, then `C E C E`, then `A D A D`, so every variant shares a session with the anchor.
`-lzm on-direct -lm none` is the reference config.

| arm | change | prediction (pre-registered) |
| --- | --- | --- |
| A | none (reference) | `lazy:prefetch` ~70 ms/call, `input:lazy_gather` ~79, `io:storage_prefill` ~36.7 MB, pp4096 ~2100-2145 |
| B | `LLAMA_LAZY_PREFETCH=0` | gather **15-30 ms** (-60..-80%), storage bytes similar or slightly lower, pp4096 **+5..+15%**. Falsifier: B within +-3% or slower, i.e. the WILLNEED loop is load-bearing and the fix has to be a bulk/staged reader instead |
| C | `-lzm off` (table read eagerly into host RAM) | gather **3-10 ms**, storage ~0, pp4096 **+10..+16%**, at a real cost in load time (record the load lines; predict +10..+60 s). Informative failure: it does not fit and falls back |
| E | `-lm mmap -lzm off` (weights mapped, no lazy reader) | gather **5-15 ms** after rep 1, pp4096 **+8..+15%**, load time close to A. This is the "resident without paying the load" arm, and it is the ceiling the user's staged prefetch would reach |
| D | `LLAMA_LAZY_PREFETCH=0 LLAMA_LAZY_WORKERS=32` and `=128` | flat to worse vs B (**-0..+3%** on the gather for 128, slower for 32). If D is broadly flat, the cold reads are already at the device's random-read rate and a one-batch-ahead *bulk* read is the right instrument (queue depth, not thread count). If 32 is much worse, latency dominates and the early issue is worth more |

**Decision rule, agreed before running:**

- B >= +5% and C/E >= +10%: ship the small change first - skip or batch the WILLNEED loop above
  `LAZY_IO_DECODE_MAX_ROWS` - then implement the one-batch-ahead issue for the read path.
- B flat or negative, C/E >= +10%: the per-row prefetch is load-bearing at this width, so the fix is a bulk
  staged read (read the rows into a bounce buffer one batch early), not a smaller fadvise loop.
- C/E flat: the gather is already overlapped and the whole idea has nothing to win on this box. Then the
  prefill lever is elsewhere (the `graph:alloc` 20.5 ms/call, or kernel work).

## harness and commands

```sh
MODEL=~/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf
BASE="-m $MODEL -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -d 16384 -p 4096 -n 0 -r 5 -b 2048 -ub 1024"

GGML_PROF_REGIONS=1 llama-bench $BASE                              # A
GGML_PROF_REGIONS=1 LLAMA_LAZY_PREFETCH=0 llama-bench $BASE        # B
GGML_PROF_REGIONS=1 llama-bench $BASE -lzm off                     # C
GGML_PROF_REGIONS=1 llama-bench $BASE -lzm off -lm mmap            # E
GGML_PROF_REGIONS=1 LLAMA_LAZY_PREFETCH=0 LLAMA_LAZY_WORKERS=32  llama-bench $BASE   # D32
GGML_PROF_REGIONS=1 LLAMA_LAZY_PREFETCH=0 LLAMA_LAZY_WORKERS=128 llama-bench $BASE   # D128
```

Keep the whole log, including the load lines (arm C and E's load cost is a result, not noise). Raw logs into
`results/user/E076-lazy-prefetch-at-prefill-width/`. One file per arm, named by the arm letter.

## correctness gate

None of these arms changes arithmetic - same file bytes, same tensor values, only the read path and the load
path move. The gate is therefore the *strongest* form: one `llama-perplexity` run on A and on the winning arm
must agree bit-for-bit at the recorded precision (E073's standard: identical to 4 decimals). If they differ,
the tensor was not what it claimed and the arm is invalid until explained.

## out of scope

- No code is changed by this round; it prices a class of fix, it does not implement one.
- The `lazy_seen` set is profiling-only and process-wide, so `io:uniq_prefill` says nothing about caching value
  across *requests*; that question needs a real served prompt pattern, not llama-bench's synthetic one.
- Device time is not measured here. `graph:compute` is an enqueue region (E065), so these arms bound the host
  side only.