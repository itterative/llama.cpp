# E058 - what the n-gram fetch costs a decode step, now that on-direct removed the faults

- date: - (armed 2026-09-25)
- machine: bench-4x-r9700-32g (T2)
- tier: T2
- status: planned
- parent: E031 (on-direct landed, +37.2% pp512 at 131k), E053 (decode kernel stats),
  [plans/ple-prefetch.md](../plans/ple-prefetch.md), backlog L1/L2/H11
- commit: `867f3eed3` (prof counters), `126b7a43b` (the instrumentation). Inert without
  `GGML_PROF_REGIONS=1`, so the build is usable for anything else
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

## notes

- The two `/proc/self/io` reads inside `gather` add ~400 B to `rchar` per call, so the miss rate is
  slightly understated: ~1.4% of a 28 KB decode call.
- `/proc/self/io` is Linux. Elsewhere `lazy_self_io` returns false and the counters stay absent.
- The counters are process-wide, not per reader, so a model with more than one lazy tensor mixes them.
  qwen4exp has one (`ple_layer_ids = [2]`).
- Bucketing is by row count at 1024, so an MTP verify of 64 tokens x 16 rows still lands in `decode`.
- If a cache does get built: it is constant-shape and therefore capture-safe (always upload
  `ple_n_heads` rows, always scatter all slots, hits rewrite identical bytes), `ggml_set_rows` is
  already in tree from H9, and storing it Q5 rather than F32 makes 1M rows 1.76 GB/card instead of
  10 GB. That is the feasibility answer, and it is not the same as the justification.
