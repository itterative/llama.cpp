# E058 - what the n-gram fetch costs a decode step, now that on-direct removed the faults

- date: 2026-09-26 (armed 2026-09-25)
- machine: bench-4x-r9700-32g (T2), run by the user
- tier: T2
- status: open - the size of the prize is measured, the fadvise arm was confounded, the reuse rate is not
- parent: E031 (on-direct landed, +37.2% pp512 at 131k), E053 (decode kernel stats),
  [plans/ple-prefetch.md](../plans/ple-prefetch.md), backlog L1/L2/H11
- commit: `867f3eed3` (prof counters), `126b7a43b` (the instrumentation), `92586c61f` (uniq/reuse
  counters). `b9301a5f1` (POSIX_FADV_RANDOM on the reader FDs) was **reverted by reset**, see below. All
  of it inert without `GGML_PROF_REGIONS=1`, so the build is usable for anything else
- model: the real Qwen3.8-Flash-Next Q4_K_M

## why now

The user runs `-lzm on-direct` and sees 1-2 MB/s of disk reads during decode. The units are
unconfirmed and they decide everything: MB/s is 8x Mbps, and the two readings imply a ~20% and a ~90%
page-cache hit rate. No existing record answers either of the questions that follow:

1. how much of a decode step is the host-side n-gram fetch, now that it lives in `set_inputs` and not
   in the graph?
2. what fraction of the fetched rows are actually cold?

## what is already known

- One trace exists (`results/user/llama-cli-traces/82bc067/llama-cli.log`, under rocprofv3):
  `graph:set_inputs` 0.6124 ms/call average, max 4.62, against `phase:decode` 47.1054 ms/call, i.e.
  **1.3%**. Two reasons not to trust it: 27 decode steps only (~40 generated tokens), so the working
  set was ~640 rows and warm after first touch; and `--kernel-trace --marker-trace` inflates the
  regions, which that log shows by reporting `graph:compute` 8221 ms total against `phase:decode`
  1271 ms total. Both cannot be wall time. Its `kernel-stats.log` looks like a different run (the
  `-o` name says `af1a347`, the directory says `82bc067`, and its call counts imply ~100x more steps
  than 27), so the two files should not be cross-read.
- Arithmetic: 16 rows/token (`ple_n_heads = (3-1)*8`) x 1760 B (Q5_0 over 2560 elems) = 28 KB useful
  per token, 64 KB if every row is its own cold 4 KiB page.
- `gather()` picks `n_workers = min(n_readers, max(1, n/32))`, so **decode (n = 16) runs on one
  worker**: 16 serial preads against 32 idle buffered FDs. Prefill (n = 16384 at `-ub 1024`) gets all
  of them. The divisor is the cheapest lever in this area and it is one line.
- E031 already removed the mid-graph CPU split: with a reader, `llm_graph_lazy_rows::build` returns an
  F32 input tensor and there is no `get_rows` node in the graph at all. So the fetch is host work
  inside `set_inputs`, and nothing synchronizes before it unless `pipeline_parallel`, so it may
  already overlap the previous step's device work.

## hypothesis

The fetch is not a meaningful share of a decode step. At 1-2 **Mbps** the rate is 1-2 cold pages per
token out of 16, a ~90% hit rate, and 0.1-0.2 ms of serial latency in a 47 ms step. At 1-2 **MB/s** it
is ~13 cold pages per token and ~1.3-1.6 ms, i.e. 3-4%.

Either way the ceiling on *any* n-gram cache, host or GPU, is the whole fetch path: under 1% in the
first case, ~4% in the second, and less after the reads are parallelised. A GPU row cache and the
divisor fix are substitutes for the same 4%, and one of them is a line.

Falsified by: `input:lazy_gather` exceeding ~5% of `phase:decode`, or `io:storage_decode` per row
running far above the 1760 B stored row size, which would mean readahead amplification after all (at
the default 128 KiB window that would be 2 MB/token, ~30x the observed rate, so this is not expected).

## conditions

Same build for every arm, same session. `-n 2000` minimum: the working set grows as 16 rows/token, so
a short run measures first touch and never reuse.

```sh
export LD_LIBRARY_PATH=$PWD/build/bin
M=<model>

# arm 1: baseline, the flags the user already runs. no rocprofv3, no GGML_PROF_DECODE
GGML_PROF_REGIONS=1 llama-cli -m $M -lm none -sm tensor -fa 1 -lzm on-direct \
  -ot per_layer_token_embd=CPU -c 131072 -b 2048 -ub 1024 -f <big prompt> -n 2000 \
  --temp 0 -s 7 -st 2>&1 | tee /tmp/e058-arm1.log
```

`GGML_PROF_DECODE` only gates the rocprofv3 capture window, so it does nothing here.

```sh
# arm 2: residency. same command, but warm the table's byte range into page cache first.
# OFFS and the row size come from the load line:
#   add_lazy_reader: ... rows of N bytes at offset OFFS of FILE
dd if=$M of=/dev/null bs=1M skip=$((OFFS/1048576)) count=$((SIZE/1048576))
```

Arm 2 is the clean residency test. `-lzm off` is **not**: with no reader the gather goes back to being
a `ggml_get_rows` CPU op inside the graph, so that arm would measure the split E031 removed rather
than residency. (This retracts a suggestion made before the code was read.)

## results

Three runs on the bench box, same command, same model, `-f prompt.md -st`, no `-n` limit. Runs 1 and 2 are
build `126b7a43b`, run 2 adding `GGML_PROF_DECODE=1`; run 3 is the fadvise arm `b9301a5f1`. Raw logs:
`results/user/llama-cli-traces/126b7a4/llama-cli.log`, `.../llama-cli-1.log`, and
`results/user/llama-cli-traces/b9301a5/llama-cli.log`.

| | run 1 | run 2 | run 3 (fadvise) |
|---|---|---|---|
| generation / prompt | 38.3 / 1289.6 t/s | 38.3 / 1293.3 t/s | 37.8 / 1268.8 t/s |
| decode gathers | 4437 | 2702 | 3982 |
| `input:lazy_gather` | 1.2423 ms/call | 1.1834 ms/call | **1.9953 ms/call** |
| `lazy_gather` max | 9.70 ms | 27.72 ms | **123.62 ms** |
| `graph:set_inputs` | 1.4503 ms/call | 1.4221 ms/call | 2.2059 ms/call |
| `phase:decode` | not enabled | 7.4786 ms/call | 7.7767 ms/call |
| `io:rows_decode` | 16/call | 16/call | 16/call |
| `io:rchar_decode` | 1.90 kB/call | 1.92 kB/call | 1.91 kB/call |
| `io:storage_decode` | 30.44 kB/call | 26.53 kB/call | **47.37 kB/call** |
| `io:rows_prefill` | 16.37k/call | 16.37k/call | 16.37k/call |
| `io:rchar_prefill` | 1.33 MB/call | 1.33 MB/call | 1.33 MB/call |
| `io:storage_prefill` | 391.58 kB/call | 42.93 kB/call | **36.15 MB/call** |
| `lazy:sort` total | 20.24 ms | 19.97 ms | 20.61 ms |
| `input:lazy_staging` total | 27.25 ms | 26.95 ms | 26.99 ms |
| `input:lazy_h2d` total | 20.40 ms | 18.76 ms | 24.05 ms |

### the size of the prize

`phase:decode` wraps all of `llama_decode` for a batch at or under the limit (`llama-context.cpp:1734`),
so it is the host side of a step only. The arithmetic closes:

    set_inputs 1.42 + graph:compute(decode) ~5.6 + rest of llama_decode ~0.46 = 7.48 ms = phase:decode
    token wall = 1 / 38.3 t/s                                                 = 26.11 ms

The missing ~18.6 ms/token is the sync, the logits read and sampling after `llama_decode` returns. So
`phase:decode` is 28% of the generation wall (20.18 s of ~71.7 s), and quoting `lazy_gather / phase:decode`
= 15.8% would overstate the prize by 3.5x. The honest share is

**`input:lazy_gather` = 1.18 ms of a 26.11 ms token = 4.5%**, and it is exposed: `set_inputs` runs before
the enqueue, at which point the device has already drained against the previous step's sync. Recovering
all of it is 38.3 -> ~40.1 t/s. That is the whole prize, and it is 8x the 1.3% the earlier rocprofv3
trace hinted at, because that run was 27 steps long and warm.

### 16 requested rows are ~1 distinct row

`io:rchar_decode` is 1.90-1.92 kB/call against a 1760 B stored row (Q5_K over 2560 elems; 32.8 GiB over
20M rows = 1761 B). Subtract the ~150 B/call of procfs overhead measured on the dummy model and what is
left is 1754 B, i.e. **exactly one row**: the 16 `ple_n_heads` indices collapse to one distinct row and
`read_range`'s coalescing turns that into one pread plus 15 memcpys of 10 kB. Prefill collapses harder:
16384 requested rows -> 1.33 MB = **756 distinct rows per 1024-token ubatch**, identical in all three runs
as it must be for the same prompt.

Not explained: `qwen4exp.cpp:1314-1323` builds 2 n-gram orders x 8 heads with per-head
`ple_head_vocab_sizes` and `ple_head_offsets`, which should give 2 distinct rows, not 1. The counters are
calibrated - on the dummy, `uniq x row_size` = 157,300 B against `rchar` 160,310 B, the difference being
the procfs overhead - so the collapse is real and not an accounting artifact. `io:uniq_*` (`92586c61f`)
now measures it directly instead of inferring it from rchar.

### ~48 kB of storage per cold row, and prefill and decode are different populations

Storage per distinct row is ~48 kB in run 3 for both buckets: prefill 36.15 MB / 756 = 47.8 kB, decode
47.37 kB / ~1. The same cold-row size fits runs 1-2 prefill, where 756 rows cost 42.93 kB (run 2: ~10 cold
pages, **98.7% hit**) and 391.58 kB (run 1: ~95 cold pages, 87% hit).

So with a warm page cache, **prefill's rows are essentially all hits and decode's single row is cold
60-100% of the time**: runs 1-2 decode at 26-30 kB/call against a ~48 kB cold row implies a ~40% hit rate,
run 3 at 47.37 kB implies ~0%. They are different row populations - the rows prefill warms are not the
rows decode asks for. That constrains every cache proposal, and it is why `io:reuse_*` is the next
measurement: only decode's own history can feed a decode cache.

Run 3 also demonstrates how sensitive this is to cache state: prefill storage fell 9x from run 1 to run 2
on byte-identical requests, purely because run 2 reused run 1's page cache.

### the fadvise arm: prediction falsified, run confounded, commit reverted

`b9301a5f1` added `posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)` to each reader FD, on the reasoning that
`llama-mmap.cpp:552` already advises `POSIX_MADV_RANDOM` over the lazy ranges while the descriptors the
reader opens itself never got the hint. Predicted: `io:storage_decode` <= 4 kB/call, `lazy_gather`
<= 0.3 ms if the row had been hitting, `rchar` unchanged as the control.

Observed: storage **up** (decode 26.53 -> 47.37 kB/call, prefill 42.93 kB -> 36.15 MB/call), gather **up**
1.1834 -> 1.9953 ms/call, tail up 27.72 -> 123.62 ms, generation 38.3 -> 37.8 t/s. The control held
exactly (`rchar` 1.92 -> 1.91 kB/call, `rows` 16/call), so the patch changed nothing about what is asked.

**Not attributable to the patch.** `io:storage_prefill / io:rchar_prefill` went from 0.032x to 27x on
byte-identical requests, and `POSIX_FADV_RANDOM` cannot un-warm a page cache. Run 3 read the same 756 rows
per ubatch that run 2 read for 42.93 kB and paid 36.15 MB for them, so the cache was cold. The run also
hit a GPU hang (RCCL without p2p), which is how a restart drops the cache. The arms differ in cache state
by ~800x and the fadvise effect is not separable from it. Reverted by reset to `7cdef26eb`; recover as
`b9301a5f1` from the reflog if the A/B below wants it back.

Two hypotheses survive for a cache-normalized A/B: (a) it was all cache state and fadvise is neutral;
(b) fadvise RANDOM genuinely costs, because a 1760 B row straddles a page boundary and without a readahead
window it takes two synchronous page reads instead of one, and at prefill a sorted 756-row gather loses
the neighbouring rows the window would have brought in for free. (b) matters because E031's +37.2% pp512
was measured with readahead on. Such an A/B must interleave arms within one session, or drop caches before
each arm (`echo 1 > /proc/sys/vm/drop_caches`), and use p2p so it does not hang.

Worth having before that A/B, because a ~48 kB per-cold-row granularity smells like large folios or a
stripe rather than readahead: kernel version, the filesystem holding the model, and that device's
`read_ahead_kb`. If the granularity is folio-sized, readahead tuning is the wrong lever entirely and
residency is the only one.

## what to read off the table

| field | meaning |
|---|---|
| `input:lazy_gather` ms/call vs `phase:decode` ms/call | the share of a decode step the fetch takes |
| `io:rchar_decode / io:rows_decode` | bytes asked per row. 1760 = every row distinct, less = duplicates coalesced by `read_range` |
| `io:storage_decode / io:rows_decode` | cold bytes per row. 0 = all warm, ~4096 = every row a cold page |
| `io:storage_decode / io:rchar_decode` | the miss rate, which is the number the MB/s-vs-Mbps question was proxying |
| `input:lazy_staging` at prefill | the zero-fill on a staging grow (336 MB at `-ub 2048`) |
| `input:lazy_h2d` | the F32 upload: 10240 B/row against 1760 stored |
| `lazy:sort` | the index sort; matters at n = 16384, expected negligible at n = 16 |

## decision table

| observation | conclusion | fix it points at |
|---|---|---|
| gather < 1% of decode | the fetch is not the cost | close L1/L2, build nothing |
| miss rate low | the page cache already does the job | close L1/L2, build nothing |
| miss rate high AND arm 2 cuts gather a lot | cold reads, residency is the lever | a cache is justified; run the index trace next |
| miss rate high AND arm 2 changes nothing | the cost is syscall/dequant, not I/O | the `n/32` divisor, not a cache |
| `lazy_staging` dominates prefill | zero-fill on grow | `reserve`, not a cache |
| `lazy_h2d` dominates | the F32 upload | upload Q5 rows and dequantize on device, not a cache |

Only the third row leads to a cache, which is the reason to measure before designing one.

**Where this landed:** no row, because the table did not anticipate the combination that appeared. The
gather is 4.5% of the token wall (not < 1%), and the miss rate is high for decode but ~1% for prefill,
which is not one "miss rate" but two. Three rows are closed as measured and dead: the `n/32` divisor
(decode reads ~1 distinct row, so there is nothing to parallelise), `lazy_staging` (27 ms total over a
~117 s run, its 4.4 ms max being the one-time 168 MB zero-fill), and `lazy_h2d` (19-24 ms total, and the
set is async so it is enqueue time). `lazy:sort` is 20 ms total. What remains is the cache row, and it now
turns on `io:reuse_decode` rather than on arm 2.

## notes

- The two `/proc/self/io` reads inside `gather` add ~150 B to `rchar` per call, measured on the dummy
  model. That is ~8% of a 1.9 kB decode call, so the miss rate is understated by that much - and
  subtracting it is what makes `rchar` come out at exactly one 1760 B row.
- The `uniq`/`reuse` pass is a linear scan of the sorted pairs plus one hash insert per distinct row,
  inside `input:lazy_gather`: ~1 us at decode (n = 16) and ~50 us at prefill (n = 16384, 756 distinct).
  Negligible at decode, worth remembering when reading prefill gather times.
- `lazy_seen` is process-wide and never evicted, so a very long run grows it. Inert unless profiling.
- `meta:subgraph` is the largest host-side number in these traces and has nothing to do with PLE: 97
  dispatches and 5.5 ms per decode step, which is essentially all of decode's `graph:compute` (5.6 ms) and
  **21% of the 26.11 ms token wall**, 4.7x the n-gram gather. `meta:allreduce` adds 1.4 ms/step. Whether
  that 5.5 ms is exposed depends on how 4-card dispatch paces against device execution, which these
  regions cannot show. Belongs to the comms thread; recorded here so it is not lost.
- `/proc/self/io` is Linux. Elsewhere `lazy_self_io` returns false and the io counters stay absent, while
  `rows`/`uniq`/`reuse` still work.
- The counters are process-wide, not per reader, so a model with more than one lazy tensor mixes them.
  qwen4exp has one (`ple_layer_ids = [2]`).
- Bucketing is by row count at 1024, so an MTP verify of 64 tokens x 16 rows still lands in `decode`.
  Run 2's `rows_decode` total (43.97k over 2702 calls = 16.27/call) shows ~46 two-token steps, so MTP does
  fire occasionally without a draft model on the command line.
- If a cache does get built: it is constant-shape and therefore capture-safe (always upload
  `ple_n_heads` rows, always scatter all slots, hits rewrite identical bytes), `ggml_set_rows` is
  already in tree from H9, and storing it Q5 rather than F32 makes 1M rows 1.76 GB/card instead of
  10 GB. That is the feasibility answer, and it is not the same as the justification.
