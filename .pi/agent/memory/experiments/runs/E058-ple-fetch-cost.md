# E058 - what the n-gram fetch costs a decode step, and the concurrency fix that took 4.2% off it

- date: 2026-09-26
- machine: bench-4x-r9700-32g (T2), every run by the user; the dev box only for calibration and gates
- tier: T2
- status: done
- parent: E031 (`-lzm on-direct` landed, +37.2% pp512 at 131k), E053 (decode kernel stats),
  [plans/ple-prefetch.md](../plans/ple-prefetch.md), backlog L1/L2/H11
- commit: `867f3eed3` (prof counters), `126b7a43b` (reader instrumentation), `92586c61f` (uniq/reuse),
  `8957f3667` (the two knobs), `3f1138bb3` (prefetch on by default). `b9301a5f1` (`POSIX_FADV_RANDOM`)
  was tried and dropped by reset; recoverable from the reflog
- model: the real Qwen3.8-Flash-Next Q4_K_M; `models/q4exp-4l.gguf` for gates and calibration

## verdict

The PLE n-gram fetch was **1.42 ms of a 26.32 ms decode token (5.4%)** because a decode step reads 16
independent 110 B rows and `gather()` put all 16 on **one thread**, so the ~8 cold ones were 8 separate
queue-depth-1 waits. Issuing `POSIX_FADV_WILLNEED` for every distinct row before waiting on any of them
takes the gather to **0.5819 ms/call** and tg **38.0 -> 39.6 (+4.2%)**. That is now the default
(`3f1138bb3`), with `LLAMA_LAZY_PREFETCH=0` to opt out.

Attribution does not rest on repetition: each arm's tg gain matches what its own gather delta predicts to
within 0.3 t/s. Along the way the measurement closed four other proposals - a GPU row cache, a host row
cache, `POSIX_FADV_RANDOM`, and the worker-count divisor - and corrected the table geometry by 16x.

## instrumentation

Three regions inside the previously undivided `graph:set_inputs`, so the host fetch path decomposes:
`input:lazy_staging` (the staging vector grow), `input:lazy_gather` (the whole gather), `input:lazy_h2d`
(the upload), plus `lazy:sort` and `lazy:prefetch` nested inside the gather. Regions nest per thread and
totals are **inclusive**, so self time is a subtraction.

`ggml_prof_count(name, delta)` (`867f3eed3`) adds named integer counters to the same report, printed
after the regions in name order. The reader uses them for `/proc/self/io` deltas taken inside `gather()`:
`rchar` is every byte asked of `pread` including page-cache hits, `read_bytes` is only what storage
served, so the pair is the miss rate. Taking them in-process is what makes them usable - an external
sampler cannot exclude model load and cannot separate prefill from decode. Calls are bucketed by row
count at 1024, so decode (16 rows) and prefill (16384 rows at `-ub 1024`) never mix.

`io:uniq_*` / `io:reuse_*` (`92586c61f`) count distinct rows per call and how many of those an earlier
gather already read. All of it is inert without `GGML_PROF_REGIONS=1`.

Calibration: on the dummy model `uniq x row_size` = 157,300 B against `rchar` = 160,310 B, the 3,010 B
difference being ~150 B/call of procfs overhead, and `reuse (256) <= uniq (1430) <= rows (1580)`.

## the table geometry, and a wrong turn worth recording

The stored row is **110 B**, one Q3_K block of 256 elements. Two independent readings agree: prefill
`rchar / uniq` = 1,333,200 / 12,114 = 110.03 B, and decode (1914 - 150 procfs) / 16 = 110.25 B. The
dummy reports the same "rows of 110 bytes". The size then closes the geometry: 20M n-gram vocab x 16
heads = 320M rows x 110 B = 35.2 GB = **32.8 GiB**, which is H12's table size.

Runs 1-3 were analysed assuming a 1760 B row (Q5_K over the 2560-wide `ple_embed_dim`), which is 16x too
big - and 16 is the head count, so every error lined up with a plausible story:

| | inferred from 1760 B | measured by `io:uniq_*` |
|---|---|---|
| distinct rows per decode token | ~1 | **16** (`uniq` == `rows` in all eight runs) |
| distinct rows per prefill ubatch | 756 | **12,114** (1.35x dedup, not 21x) |
| storage per cold row | ~48 kB | **one 4 kB page** |

Two conclusions flipped because of it, in opposite directions: the worker divisor looked dead (one row,
nothing to parallelise) and is in fact the mechanism, and the readahead-amplification story looked real
(48 kB per row) and is in fact absent (9.8 pages for 16 rows). **Only the `uniq` counter caught it.** A
bytes-per-call counter cannot distinguish "few rows, big amplification" from "many rows, no
amplification", and both fit the same 1-2 MB/s the user had observed by eye.

All 16 heads want different rows, which the index code supports: `mixed % ple_head_vocab_sizes[h] +
ple_head_offsets[h]` (`qwen4exp.cpp:1314-1323`) puts each head in its own slice, so they cannot collide.

**The correct geometry was already written down.** `plans/qsa-ple-in-prefill-and-decode.md:43` says
"16 rows of 110 bytes per token = 1760 B", and the error was reading that as *one* 1760 B row. Three runs
were analysed on it before a counter contradicted it, so the lesson is to check a derived per-token byte
total against the per-access count whenever the mechanism turns on accesses rather than bytes.

## runs 1-4: sizing the cost

Same command, `-f prompt.md -st`, default sampling, so each run generated different text.

| | run 1 | run 2 | run 3 (`b9301a5f1`) | run 4 (`92586c61f`) |
|---|---|---|---|---|
| tg / pp | 38.3 / 1289.6 | 38.3 / 1293.3 | 37.8 / 1268.8 | 37.8 / 1292.7 |
| decode gathers | 4437 | 2702 | 3982 | 2820 |
| `input:lazy_gather` | 1.2423 ms | 1.1834 ms | 1.9953 ms | 1.5976 ms |
| gather max | 9.70 ms | 27.72 ms | 123.62 ms | 51.77 ms |
| `graph:set_inputs` | 1.4503 ms | 1.4221 ms | 2.2059 ms | 1.8301 ms |
| `phase:decode` | not enabled | 7.4786 ms | 7.7767 ms | 7.8199 ms |
| `io:rchar_decode` | 1.90 kB | 1.92 kB | 1.91 kB | 1.91 kB |
| `io:storage_decode` | 30.44 kB | 26.53 kB | 47.37 kB | 40.02 kB |
| `io:storage_prefill` | 391.58 kB | 42.93 kB | 36.15 MB | 2.71 MB |

`phase:decode` wraps all of `llama_decode` for a batch at or under the limit (`llama-context.cpp:1734`),
so it is the **host side of a step only**: `set_inputs` 1.42 + decode's `graph:compute` ~5.6 + ~0.46 =
7.48 ms, against a 26.11 ms token wall. The other ~18.6 ms is the sync, the logits read and sampling after
`llama_decode` returns. Quoting `gather / phase:decode` = 15.8% would overstate the share 3.5x; the
honest figure is gather over the token wall, 4.5-6.0% across these runs.

Prefill is ~99.9% warm whenever the cache holds: run 2 read 12,114 distinct rows per ubatch for 42.93 kB
of storage, about ten cold pages. Run 3 is the exception at 36.15 MB (73% cold) and it also hit a GPU
hang, so its cache had been dropped - which is why its fadvise result is not attributable, see below.

The user's "1-2 MB/s during decode" was MB/s and it reproduces: run 4 read 112.87 MB over ~74.6 s of
generation = 1.51 MB/s.

## run 5: the four-arm A/B (`8957f3667`)

Interleaved on one binary, one pass each. `--temp 0` was tried first and **made the model loop**, and
`-s <seed>` did not reproduce across runs, so the arms generated different text - 2863 to 4747 decode
steps. Per-call normalization is what makes them comparable.

| arm | decode calls | rows | uniq | rchar | storage | cold pages | gather | **us/cold page** | gather min | tg |
|---|---|---|---|---|---|---|---|---|---|---|
| A baseline | 2863 | 16.26 | 16.26 | 1914 B | 33.39 kB | 8.15 | 1.4181 ms | **174** | 0.05 | 38.0 |
| B `PREFETCH=1` | 4747 | 16.15 | 16.15 | 1902 B | 30.49 kB | 7.44 | **0.5819 ms** | **78** | 0.06 | **39.6** |
| C `WORKERS=8` | 4096 | 16.18 | 16.18 | 1907 B | 29.86 kB | 7.29 | 0.7929 ms | 109 | **0.29** | 39.1 |
| D both | 4373 | 16.17 | 16.17 | 1905 B | 26.41 kB | 6.45 | 0.7148 ms | 111 | 0.31 | 39.2 |

Controls held. `uniq == rows` in every arm, and `rchar` is within 0.6% (1902-1914 B) because it is
structurally 16 x 110 B plus procfs regardless of *which* rows. Prefill counters are **byte-identical
across all four arms** (`rchar` 33.33M, `rows` 409.30k, `uniq` 302.86k, `reuse` 91.64k) because
`prompt.md` is the same, and prompt rate is 1292.0 / 1290.1 / 1292.3 / 1292.2 t/s - within 0.15%, so no
arm costs anything at prefill. Only cold-ness drifted, 26.4-33.4 kB/call, so it is divided out: latency
per cold page falls **174 -> 78 us** in B, a 2.2x concurrency win against a cache-state spread of 1.27x.

B's 0.5819 ms includes the nested `lazy:prefetch` 0.1207 ms, so its read self-time is 0.4612 ms = 62 us
per cold page. C and D show a hard floor at `gather min` 0.29/0.31 ms against A/B's 0.05/0.06 ms - that
is the cost of spawning 7 threads per token, and it is why C loses despite reading as fast as B.

tg accounting, from A's 26.316 ms/token:

| arm | gather saving | predicted | measured |
|---|---|---|---|
| B | 0.836 ms | 25.480 ms -> 39.25 t/s | **39.6** |
| C | 0.625 ms | 25.691 ms -> 38.92 t/s | 39.1 |
| D | 0.703 ms | 25.613 ms -> 39.04 t/s | 39.2 |

Every arm lands within ~0.3 t/s of its own prediction, so the gain is the gather and nothing else.

Prefetch cost 0.1207 ms/call for 16 rows on the bench box (7.5 us per `fadvise` that starts real
readahead, against 0.75 us when the page is already cached, which is what prefill's ~9 ms per 12,114-row
ubatch implies).

## closed

- **Any row cache, host or GPU.** The misses are **compulsory first touches**, not evictions: 2820 tokens
  introduced 33.2k distinct rows = **133 MB of pages**, a trivial working set on a 62.7 GiB box, so
  nothing is being squeezed out. `io:reuse_decode` is 27.7-34.1% and those rows are *already free* via
  the page cache, which is why storage is 26-33 kB/call and not the 64 kB/call of 16 cold pages. A cache
  can only re-capture what the kernel already captures, and a page-cache hit is ~1-2 us. The feasibility
  work in `ple-prefetch.md` (constant shapes so capture survives, `ggml_set_rows` already in tree from
  H9, Q5 storage at 1.76 GB per million rows) stands, but there is nothing for it to win.
- **`POSIX_FADV_RANDOM`.** 9.8 pages for 16 rows means at most one page per row, so there is no readahead
  amplification to remove. Run 3's higher numbers were its cold cache: `storage_prefill / rchar_prefill`
  went 0.032x -> 27x on byte-identical requests, which fadvise cannot cause. Dropped without an A/B.
- **The `n/32` worker divisor.** Alive again once `uniq` showed 16 distinct rows, then measured: it works
  (174 -> 109 us/page) but pays ~0.29 ms/token of thread spawn, so prefetch dominates it. The knob stays
  for tuning; the default formula is unchanged.
- **`lazy_staging`, `lazy_h2d`, `lazy:sort`.** 27-30 ms, 19-24 ms and 20 ms total over a ~75-120 s run.
  The staging max of 4.4 ms is the one-time zero-fill of a 16.8 MB grow at `-ub 1024` (16384 rows x 256
  elems x 4 B), not a per-step cost. The upload is async, so `lazy_h2d` is enqueue time.
- **`-lzm off` as a residency test.** With no reader the gather becomes a `ggml_get_rows` CPU op inside
  the graph again, so that arm measures the split E031 removed rather than residency.

## open

- **The tail.** `lazy_gather` max is 28.97 / 30.29 / 30.41 / 30.63 ms across the four A/B arms - one row
  read costing more than a whole token's budget, in every arm including the fixed one, and 51.77-123.62 ms
  in runs 3-4. Prefetch cut the mean 59% and barely touched the max. Whatever that is, it is not
  queue-depth-1 latency, and it is the irregular-tg signature backlog L1 predicted.
- **`meta:subgraph` is now the biggest host-side number**: 97 dispatches and ~5.4 ms per decode step,
  essentially all of decode's `graph:compute` and ~21% of the token wall, which is 3.4x the *fixed*
  gather. `meta:allreduce` adds ~1.4 ms/step. Comms thread, not PLE.
- **Warm-cache cost of the new default.** When rows are already cached, prefetch is pure overhead: the
  dev box measured gather 0.1194 -> 0.1687 ms/call (+41%) with it on. On the bench box that is ~0.12 ms
  of a 25.3 ms token, ~0.5%, against +4.2% when the table cannot stay cached. `LLAMA_LAZY_PREFETCH=0`
  opts out. Whether a size threshold is worth the magic number is untested.
- **Why `-s <seed>` does not reproduce** across runs with `-sm tensor`. The likely cause is
  non-deterministic 4-card reduction order moving the last bits of the logits, not the sampler. If so, no
  llama-cli run on this box can be text-matched and every A/B has to normalize per call. Unconfirmed.

## notes

- The two `/proc/self/io` reads inside `gather` add ~150 B/call to `rchar`, ~8% of a decode call, so the
  miss rate is understated by that much. Subtracting it is what makes `rchar` come out at 16 x 110 B.
- `lazy_seen` is process-wide and never evicted, so decode's reuse figure includes rows *prefill* read.
  Decode-only reuse is at most the measured 27.7-34.1%, which only strengthens the cache conclusion.
- Cold pages are derived as `storage / 4096`. If the device granularity is not 4 kB the absolute count is
  off, but every arm ran on the same device, so the ratios hold.
- `/proc/self/io` is Linux; elsewhere `lazy_self_io` returns false and the io counters stay absent while
  `rows`/`uniq`/`reuse` still work. The counters are process-wide, not per reader, so a model with more
  than one lazy tensor mixes them; qwen4exp has one (`ple_layer_ids = [2]`).
- Run 3 is kept even though it is not attributable: it is the only observation of a cold-cache prefill
  (73% cold, 36.15 MB per ubatch) and of the 123.62 ms gather tail.
- Raw logs are `.log` under `results/user/`, uncommitted by PROTOCOL 8 and regenerateable from the command
  in each file's first line.
