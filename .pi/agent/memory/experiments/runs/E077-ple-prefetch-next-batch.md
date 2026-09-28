# E077 - prefetch the next batch's n-gram rows: implemented, exact on the dev box, the box is the case for it

- date: 2026-09-28
- machine: dev-rx9070-16g (warm table); bench-4x-r9700-32g carried the cold case
- tier: T1 (implementation + dev box) and T2 (bench box, one sweep)
- status: done
- parent: E058 (the WILLNEED primitive, the 110 B row, the table geometry, and what it closed), E065/E075
  (post-H19 the dev box's prefill is device-bound), E074 (the box's gather: 79.2 ms/call, 36.75 MB from
  storage per ubatch, 73% cold)
- code: this run's patch is in the tree, uncommitted at time of writing
- raw: scratchpad `lz-*.log`, `lz48-*.log`; the box's logs go to `results/user/E077-*/` (gitignored)

## question and what decides it

E074's box numbers put a cold, per-batch I/O burst inside `graph:set_inputs`: 16384 distinct rows, 1.80 MB
requested, **36.75 MB fetched from storage** (a 4 KB page per 110 B row), 70.8 ms of it the per-row WILLNEED
loop. It sits between `graph_compute(k-1)` returning and `graph_compute(k)` being enqueued, so on a host that
serialises with the device it can overlap nothing. E058 had already closed the instruments for this table (no
row cache, no `POSIX_FADV_RANDOM`, no worker-divisor tuning) and measured its own prefill as ~99.9% warm,
because that harness re-used one prompt.

Design implemented, from the user's proposal: prefill knows the tokens that follow, so issue the *next*
window's WILLNEED calls while the current batch computes.

1. `llama_lazy_reader` gets a lazily started background thread and a 1-deep job slot. `prefetch(rows, n)`
   sorts/uniques and hands the list over; the thread runs the same WILLNEED loop. It reads no data and writes
   no buffer, so it cannot race the gather that follows - worst case that gather waits on an in-flight page.
2. `gather()` compares its own distinct-row list against the last prefetched list and skips its WILLNEED loop
   on a match (70.8 ms cold against ~12 ms cached, per E058's 4.3 us vs 0.75 us per row). That compare also
   doubles as the exactness probe for the lookahead.
3. `llama_context::decode`'s ubatch loop finds the window that follows the current ubatch in the allocator's
   batch (positions, sequence and token must continue it) and calls the model.
4. `llama_model::prefetch_next_rows` is a no-op virtual; qwen4exp overrides it and rebuilds each token's
   n-gram window from tokens alone (carry = the closing ubatch's tail, then the window's own earlier tokens),
   sharing `ple_mix_row` with the exact path in `set_inputs`. Advisory: a wrong window only wastes pages.

Deciding metric: pp t/s against the pre-registered prediction, with `input:lazy_gather` and `graph:set_inputs`
ms/call, `io:storage_prefill` bytes/call, and the new `io:prefetch_rows` + `io:ahead_hit/miss_*` counters as
the mechanism evidence. Prediction, written before the dev runs: covered gathers drop to ~14 ms, coverage is
`(n_ubatches-1)/n_ubatches` per decode call, and pp gains **+10..16%** only if the gather is on the critical
path - flat if the host is already overlapped.

## arms and results, dev box

`llama-bench -fa 1 -ngl 99 -lzm on-direct -b 2048 -ub 1024`, `GGML_PROF_REGIONS=1`, 3 runs each. `-lzm
on-direct` puts the PLE table behind the reader (it is `TENSOR_READ_LAZY`).

| arm | 4l `pp8192` | 4l `set_inputs` | 48l `pp4096` | 48l `set_inputs` |
| --- | --- | --- | --- | --- |
| ahead (default) | 14997.05 +- 38 | 19.69 ms/call | 1743.72 +- 5 | 15.89 ms/call |
| `LLAMA_LAZY_PREFETCH_AHEAD=0` | 15009.48 +- 60 | 25.48 ms/call | 1742.33 +- 1 | 20.45 ms/call |
| `LLAMA_LAZY_PREFETCH=0` | 15275.86 +- 51 | 14.13 ms/call | 1745.89 +- 2 | 11.26 ms/call |

Mechanism, 48l ahead arm: `io:prefetch_rows` 6 calls / 98.30k rows, `io:ahead_hit_prefill` **6**, miss 6,
`input:lazy_gather` 11.67 ms/call over the 12 gathers. 4l: 12 issued, 12 hit, 0 miss. So:

- **the window rebuild is exact** - every prefetch issued was matched by the gather that followed, with zero
  wasted pages, on two different models (48 layers/12 QSA and 4 layers).
- coverage is exactly the predicted 50% at `-b 2048 -ub 1024`: the second ubatch of each decode call is
  covered, the first cannot be (the tokens of the next *call* are unknown). Per decode call it is
  `(n_ubatches-1)/n_ubatches`.
- the covered gathers lose their WILLNEED loop, which is why `set_inputs` falls 5.8 / 4.6 ms per call on
  average while `lazy:prefetch` stays flat: the same syscalls, moved off the main thread.

**pp does not move** (14997 against 15009, 1743.7 against 1742.3; removing all prefetch gives +1.8% / +0.2%).
That is the honest negative of this round and it has a cause: on this box `io:storage_prefill` is **0 in every
arm** - the rows are page-cache resident (5.8 and 11.9 GB tables, 80 GB of buff/cache) - so the gather is only
syscall time, and syscall time is already hidden behind device work. E075 already showed the same box's
post-H19 prefill is 93% inside `graph:compute`. The dev box cannot express this experiment's target case.

## correctness

- PPL bit-identical to the golden value on the default path and on `-lzm on-direct`: **263100.7437** both.
  The patch touches no arithmetic: the reader still reads the same rows and still dequantizes with the same
  `to_float`.
- decode untouched: `-p 512 -n 128` gives `input:lazy_gather` 0.044 ms/call over 384 calls and
  `io:ahead_miss_decode` 384/384, i.e. no lookahead exists at 1-token width and E058's decode path is
  unchanged. `tg128` 278.60 +- 2.93 on the 4l.
- The prefetch thread never calls `ggml_prof_count` or opens a region: the prof tables are plain globals with
  no lock, so the region moved to the main-thread call site in `gather()`.

## knobs

- `LLAMA_LAZY_PREFETCH_AHEAD` (new, on by default): `=0` keeps the WILLNEED loop inline in `gather()` only.
- `LLAMA_LAZY_PREFETCH` (existing): `=0` disables the prefetch entirely, inline and ahead.

## open - the box

The bench box is the only machine that expresses it: 36.75 MB of *storage* per ubatch, 70.8 ms of WILLNEED,
and 4-card serialisation (E074's phases sum roughly to the wall). One run each way answers it:

```sh
B="-m <model> -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -d 16384 -p 4096 -n 0 -r 10 -b 2048 -ub 1024"
GGML_PROF_REGIONS=1 llama-bench $B                                  # ahead
GGML_PROF_REGIONS=1 LLAMA_LAZY_PREFETCH_AHEAD=0 llama-bench $B      # control
```

Read `pp4096`, `input:lazy_gather` and `io:storage_prefill`, plus `io:prefetch_rows` and
`io:ahead_hit/miss_prefill` to check the lookahead is exact there too.

- If pp gains and the covered gathers lose their read waits, the gather is on that box's critical path and the
  patch is the fix it was written to be (E074's 79.2 ms/call against ~479 ms per ubatch bounds the prize at
  ~+10-14%).
- If pp is flat there too, the box's prefill is dispatch/device-bound as well, the I/O is already overlapped,
  and the prefill lever is the `graph:alloc` 20.5 ms/call and the 6825-node dispatch (E074's other half) -
  worth knowing before anyone spends more on the table's read path.

## bench box result (T2, one sweep, cross-session)

User run at `02dd5cea2` against the recorded reference at `ede6d511c`: `-lm none -sm tensor -fa 1 -lzm
on-direct -ot per_layer_token_embd=CPU -d 4096,16384,40960,131072 -p 512,4096,8192 -n 128 -r 10 -b 2048
-ub 1024`, raw in `results/user/e79-02dd5cea2/run.log` against `results/user/h19-runs-ede6d511c/ref-run.log`.
The H19 fix is neutral on this box (E074), so the delta is this patch; prefill is the phase that is stable
across sessions (project memory), which is what makes the comparison readable.

| test | d4096 | d16384 | d40960 | d131072 |
| --- | --- | --- | --- | --- |
| pp4096 | **+3.03%** (2125.42 -> 2189.79) | **+1.91%** (2057.11 -> 2096.36) | **+1.18%** (1908.85 -> 1931.34) | +0.02% |
| pp8192 | +1.81% (2201.86 -> 2241.66) | +1.02% | +0.60% | -0.06% |
| tg128 | +0.21% | +0.14% | +0.21% | -0.03% |
| pp512 | +8.34% | +6.24% | +4.30% | +1.32% |

What it says:

- **The gather is only partly exposed on this box.** At d4096 pp4096 the wall per rep is 1.87 s and the
  covered windows carry 2 x 79 ms of gather, of which the patch recovers about 55 ms - so ~27 ms per covered
  window was on the critical path and the rest was already overlapped by the kernel's readahead. That is the
  same shape the dev box showed, just with real cold pages underneath.
- **The exposed part is a fixed per-ubatch cost, so depth dilutes it.** +3.0% -> +1.9% -> +1.2% -> +0.0% as
  the per-ubatch device work grows; at 131k there is enough of it to hide the whole gather. Decode is flat,
  as designed (no lookahead exists at 1-token width).
- pp512's numbers are not usable evidence: its own spread is +-60..196 t/s (13%), larger than the delta.
- Cross-session comparison against `ede6d511c` with no in-session control: the deltas rest on prefill being
  the cross-session-stable phase (project memory) and on the sweep's own spread (pp4096 +-2.6..3.4 t/s),
  which is far smaller than the effects claimed. Interleaving the control arm would harden it further.
- **Counter evidence, from a second run in the same session** (`run-prof.log`, `GGML_PROF_REGIONS=1`,
  `-d 16384 -p 4096 -n 0 -r 10`, pp4096 2095.50 +- 4.55 against the sweep's 2096.36 +- 2.62, so the
  instrument costs nothing): `io:ahead_hit_prefill` **28** and miss 28 over 56 gathers - the predicted 50%
  coverage - and `io:prefetch_rows` 28 calls / 458.75k rows, i.e. **every window issued was matched, zero
  wasted pages on the real model**, as on both dev models. `input:lazy_gather` fell from E074's 79.2 to
  **43.54 ms/call**, which at 50% coverage means a covered window costs ~8 ms and an uncovered one ~79:
  **~71 ms leaves the host path per covered window**. Only ~19 ms of it reached the wall at d16384 (37 ms
  per rep over 2 covered windows) and by the same arithmetic ~28 ms did at d4096 (+3.0%), so the host was
  39% exposed there against 27% at d16384 and hidden at 131k - the depth trend in one number.
  `io:rchar_prefill` is unchanged at 1.80 MB/call and `io:uniq_prefill` still equals `io:rows_prefill`
  (16.38k, no repetition inside a call), so the row list did not change, only when its reads are issued.
  `io:storage_prefill` is 28.35 MB/call against E074's 36.75; this run followed another sweep in the same
  session, so a partly warm cache is the likely cause and it is left unattributed.
- The user's read, recorded: llama-bench's random token fill makes every row a compulsory miss, so it cannot
  show what a workload with reused context would see. Quantified: a warm table still saves the ~12 ms/call
  WILLNEED loop on covered windows (workload-independent), and a fresh long document pays the cold part that
  llama-bench already models - so a server with reused context should land towards +1%, a first sight of a
  document towards +3%, and a deep context towards nothing because it is hidden there anyway.

Net: landed as implemented, worth **+1.2..+3% pp** on this box at the depths anyone works at, flat at depth
and flat in decode. The staging-fill extension (thread does the reads and `to_float`, `set_rows` collapses to
one upload) was offered and declined as unnecessary for this size; the H2D it would also cover is 0.72 ms/call,
0.2% of that box's prefill.

## notes

- What stays closed from E058: no row cache, no `POSIX_FADV_RANDOM`, no worker divisor change. This patch adds
  no cache and changes no read granularity - it moves *when* the same WILLNEED calls are issued.
- `io:uniq_prefill` == `io:rows_prefill` is not a near-miss on the 4l either (393.22k rows, uniq within 0.2%),
  so within a prefill the 16 rows per token are all distinct, as E058 found.
- The reader thread is joined in the reader's destructor; a job still pending when a new one arrives is
  dropped, which by construction is the window before it.