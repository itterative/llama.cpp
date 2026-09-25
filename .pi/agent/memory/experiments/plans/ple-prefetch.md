# PLE / n-gram table: prefetch design notes

## Update 2026-09-25: E031 changed the mechanism, E058 is the measurement

Everything below was written when the table was demand-paged through the mmap. `-lzm on-direct`
(`84b141ac6`, [E031](../runs/E031-lazy-direct-reads.md)) replaced that: `gather()` dedupes and sorts an
ubatch's row indices and reads them with positional `pread`s on buffered FDs, dequantizing to F32
host-side. What that does to this plan:

- **There are no page faults left to count**, so methods 2, 3 and 5 below measure the wrong thing.
  Method 1 (`iostat`) still works but is superseded: `126b7a43b` reports `/proc/self/io` deltas around
  `gather()` through the prof counters, bucketed decode vs prefill, which attributes the bytes exactly
  and cannot be contaminated by model load. That is [E058](../runs/E058-ple-fetch-cost.md).
- **The mid-graph split is already gone.** With a reader, `llm_graph_lazy_rows::build` returns an F32
  input tensor and there is no `get_rows` node in the graph, so section (b)'s "removes the mid-graph
  sync" prize no longer exists. What a VRAM cache would still save is the preads on hits, the host
  dequant, and the F32 upload (10240 B/row against 1760 stored).
- **`-lzm off` is not a residency test.** With no reader the gather becomes a CPU `get_rows` inside the
  graph, so that arm measures the split, not the cache state. The clean residency arm keeps `on-direct`
  and warms the table's byte range with `dd` first. This retracts the "(a) alone" bullet below as
  written, and the `--load-mode mmap+mlock` A/B with it.
- **The VRAM cache got more feasible and less justified.** Feasible: `ggml_set_rows` is in tree (H9),
  constant shapes are achievable so graph capture survives, and Q5 storage makes 1M rows 1.76 GB/card
  instead of 10 GB. Less justified: the ceiling is now arithmetic - 16 rows x 1760 B = 28 KB/token, so an
  all-cold decode pays ~1.6 ms of a 47 ms step - and `gather()` reads decode's 16 rows on **one worker**
  (`n_workers = min(n_readers, max(1, n/32))`), so most of that 1.6 ms is recoverable with one line
  instead of a subsystem. E058 decides which.

**E058 ran (2026-09-26).** The gather is **4.5% of the token wall** (1.18 ms of 26.11 ms at 38.3 t/s),
not the ~10% this note's ceiling allowed and not the 1.3% the one earlier trace hinted at. Two of the
assumptions above were wrong in ways that matter: a decode step requests 16 rows but reads **one** distinct
row of 1760 B, so "16 faults per token" was 16x too high; and a cold row costs **~48 kB of storage**, not
4 kB, so the byte side was ~12x too low. Those roughly cancel in the total, which is why the ceiling landed
close. What does not cancel is the split: prefill's 756 distinct rows per 1024-token ubatch are ~99%
page-cache hits while decode's one row is cold 60-100% of the time, so "(a) alone - warm the table" cannot
be judged from a prefill-heavy run, and the (b) VRAM row cache would be fed only by decode's own history.
Whether that history repeats is `io:reuse_decode` (`92586c61f`), and it is the last open question here.

Killed: **E007** (put the table in VRAM). Reason, from code not taste - under `-sm tensor`
the PLE table is **mirrored, not split** (`src/llama-model.cpp:513-515`,
`GGML_BACKEND_SPLIT_AXIS_MIRRORED`, because "its conv is mirrored, so every device runs the
whole conv and needs the whole history"). A ~30 GB table becomes ~30 GB *per card* = ~120 GB
of a 128 GB box, leaving nothing for KV at 40 k context. The QSA indexer cache is mirrored for
the same reason (`:508-510`), which is also worth remembering when sizing anything on this box.

What survives of F3 is narrower and still the prime suspect for tg: not *where* the table
lives, but that the **gather runs on the host and splits the graph at layer index 1 every
step** (`-ot per_layer_token_embd=CPU` forces the CPU buft; `weight_buft_supported` builds the
hypothetical `get_rows` on that buft, `src/llama-model-loader.cpp:946-950`, so the op and its
H2D result land off the GPU path).

## Status: a thing to look at, not a plan

The user's position, which is the honest one: it is **hard to say what the PLE-side issue
actually is**. The working assumption is that not mmapping the table properly may be causing
it, but that is a suspicion, not a diagnosis, and nothing below should be read as agreed.
An **LRU in RAM** would also work, and is strictly less invasive than the VRAM variant - see
"where a cache should live" at the end. Measure first (next section), then pick.

## An arithmetic ceiling that argues against this being the tg cost

Before building anything, note how little the PLE fetch path can explain. Per decoded token:
16 rows x 2560 elements at Q5-ish packing is ~1760 B per row, so ~28 KB total, spanning
roughly 16-23 page touches.

- all minor faults, page already in cache: single-digit us each -> **~0.1 ms/token**
- all cold major faults from NVMe with readahead off (~100-200 us each): **~2-4 ms/token**

Measured tg is 35.5 ms/token. So even the pessimistic end is ~10% of the cost, **unless the box
is swapping** - which is exactly what method 1 below distinguishes in five seconds. Conclusion:
the PLE fetch path is unlikely to dominate decode, and this line of work should be justified by
measurement rather than by the plausibility of the story. It may still matter for pp (where the
whole prompt's rows are touched in one pass) and for VRAM/RAM sizing, which are different
questions from tg.

## How to measure page faults here, cheapest first

| # | method | code needed | what it answers |
|---|---|---|---|
| 1 | **`iostat -x 1` / `/proc/diskstats` sampled during a steady tg run** | none | decisive and immediate: if the NVMe is serving reads while decoding, the table is being re-faulted from disk. If disk reads are ~0, the pages are resident and the whole fault-latency theory is wrong and the suspect is the sync, not the bytes |
| 2 | `perf stat -e minor-faults,major-faults -p <pid>` across a fixed token count, then faults/token | none | per-process fault counts. Major faults are the expensive ones; if major ~ 0, cost per token is bounded by minor-fault handling (~sub-us) and cannot be 35 ms |
| 3 | `bpftrace` on `tracepoint:exceptions:page_fault_user`, or on `mm_filemap_add_to_page_cache` (fires only when a file page is brought in from disk), with a stack probe | one-liner, no rebuild | attributes faults to the gather path instead of guessing |
| 4 | **`mincore()` over the table's byte range**, or `/proc/<pid>/pagemap` for the same range | small patch, or an external reader of the mapping's address | the only method that answers residency *of that specific range*: how many of the ~30 GB are present after N tokens, and whether it decays over time. llama.cpp already knows the exact ranges (`lazy_ranges`, `llama-model-loader.cpp:1096-1098`), so a debug print is a few lines |
| 5 | `clear_refs` on the mapping then re-measure faults per token | small patch | turns "I think we fault 16 times per token" into a number, with the residency reset deliberately |

Start with 1: it is free, it takes one decode run, and it can close the question in either
direction before any code is written. Add `vmstat 1` for swap in/out while there - the ceiling
arithmetic above only holds if the box is not swapping, and if it *is*, every estimate here has
to be redone.

## Where a cache should live

Given the gather is **already host-side** (`ggml_get_rows` on a CPU-placed weight, hash
computed on the host because ggml has no int64/xor):

- **A RAM-side cache is the smaller change.** It needs no new ggml op, no index remapping, no
  interaction with graph capture - it is a host data structure in front of the file-backed
  range, i.e. effectively "pin the hot rows". If the problem is fault latency and RAM is
  available for the hot set, this is the fix, and it is compatible with keeping `-lm none`.
- **A VRAM cache is the bigger change**: a new on-device tensor, remapped row indices, miss
  paths back to the host, and it is exactly the kind of dynamic-shape thing that may itself
  defeat graph capture - which is one of the things we suspect is already wrong.
- Caveat on the RAM idea: if the binding constraint is *system RAM pressure* (which is why
  `-lm mmap` thrashes at all), a RAM cache competes for the same scarce resource and wins only
  by being smaller and smarter about eviction. So steps 1 and 4 above are what determine
  whether a RAM cache is a fix or just a re-labelled version of the same problem.

## The shape of the problem, which the user's two ideas split cleanly

Facts that matter, all from `plans/model-shape.md`: PLE serves **exactly one** layer
(`ple_layer_ids = [2]`), `ple_n_heads = (ngram_size-1) * heads_per_ngram = 2*8 = 16` rows
gathered per token, row width 2560.

| mode | rows per step | bytes (f16 result) | index availability |
|---|---|---|---|
| pp, ubatch 512 | 512 x 16 = 8192 | ~42 MB | **fully known before the graph runs** |
| pp, ubatch 2048 | 32768 | ~168 MB | ditto |
| tg, 1 token | 16 | ~82 KB | known at step start, but see below |

So the table's *bandwidth* cost is trivial in every case: ~168 MB at worst, against ~48 GB/s
effective H2D per card. **The cost must be latency** - page faults, and the sync implied by a
mid-graph split. That reframing is the useful output of this note: nothing here needs more
bytes, everything here needs fewer stalls.

## (a) Prefetch into RAM - the user's first idea

Their constraint is real and not negotiable: `-lm mmap` on a 119.6 GB file thrashes the page
cache at load on a box that cannot hold it, which is *why* they run `-lm none`. So the fix is
not "use mmap"; it is to warm **only the table's range**.

That is close to a one-liner, and the machinery is already there. `init_mappings()` maps the
file whenever `lazy.any()` even with `use_mmap == false`
(`src/llama-model-loader.cpp:1411-1412`, with a comment saying lazy is deliberately usable
without `--load-mode mmap`), and the lazy byte ranges are already collected into
`lazy_ranges` (`:1096-1098`). The prefetch loop in `src/llama-mmap.cpp:499-502` then iterates
`ranges_complement(lazy_ranges, ...)` - i.e. it deliberately advises **everything except**
the table - and the table ranges get `POSIX_MADV_RANDOM` (`:508-510`), which switches kernel
read-ahead *off* for them. `MAP_POPULATE` is skipped for the same reason (`:481`).

So: an opt-in `advise(range, POSIX_MADV_WILLNEED)` **over** `lazy_ranges` - or a `-lzm warm`
mode that does exactly that - warms ~30 GB into page cache at load while leaving the
sequential-pread path for the other ~90 GB untouched. No interaction with the mmap-vs-none
decision they already made.

Worth noting the design intent being worked around: `MADV_RANDOM` is the right hint for a
random gather *within a resident range*, and the current code assumes the range will not be
resident. F1 is the gap between those two assumptions.

## (b) Prefetch into VRAM - the user's second idea, and where it stops working

Good idea, with an asymmetry between the two modes that determines what it can buy.

**For pp it should work well.** The row set for an entire ubatch is computable before the
graph runs: the tokens are known, and predecessors come from the attention KV cells
(`ext.tok`, per the PLE input comments in `src/models/qwen4exp.cpp`), which are already filled
for the prompt's history. The hash itself is host-side already (no int64/xor in ggml), so a
pre-pass could compute indices, gather once, and issue one async H2D that overlaps layer 0 -
turning ~16 faults plus a mid-graph sync into a single batched transfer per ubatch. It also
removes the *per-layer* problem entirely, since there is only one PLE layer.

**For tg it cannot pre-know the row set.** At step t the rows depend on tokens
(t-2, t-1, t). All three exist at step start, so the indices are computable - but the
consumer is layer **index 1**, i.e. essentially the first thing that runs, so the overlap
window is one layer deep and the copy is tiny (~82 KB). Prefetching is therefore racing, not
pipelining. What you *cannot* do is prefetch for t+1, because token t+1 does not exist until
the sampler returns. So for decode, (b) buys little on its own.

For decode the options that do look promising are different in kind:

- **(a) alone** - if the whole table is warm in page cache, the 16 faults per token become
  minor faults, which may be all this needs. Testable with zero code change by warming the
  file externally before a run: `vmtouch`/`dd` over the table's byte range, or a temporary
  build tweak. Cheap enough to be the first thing to try after E008.
- **A VRAM row cache with locality - endorsed by the user as the best option.** Real text is
  wildly non-uniform over n-grams, so a few-hundred-MB LRU of recently used rows on each card
  should carry a high hit rate in steady decode, with misses falling back to the host path.
  Design notes:
  - **policy: start simple.** (The user's "s2" turned out to be **S3-FIFO**, confirmed.) Our
    suspected problem is that the host is in the decode critical
    path, so per-access bookkeeping cost matters as much as hit rate. ARC's ghost lists and
    two-sublist promotions are real CPU work; FIFO-family policies (SIEVE, S3-FIFO) get
    comparable-to-ARC hit rates with far less metadata. Related lesson: FASTD showed a simple
    policy placed well can beat a complex policy placed badly. Candidates: LRU, SIEVE
    (NSDI 2024), S3-FIFO (SOSP 2023), LIRS (classic, strong on scan-ish traces), ARC.
  - **batching likely matters more than policy.** 16 rows per token = 16 lookups; compute the
    miss mask and issue **one coalesced H2D** for the misses. Since the table is mirrored,
    keep one private cache per card and no cross-card traffic.
  - **a cache does not remove the host-side hash.** Indices must be built on the host every
    step either way (no int64/xor in ggml), so this fixes the *data fetch*, not the compute or
    the sync. Payoff therefore depends on which of the three costs the time - unmeasured.
  - **decide it from data, not argument.** Log the row indices from one representative decode
    run, then simulate every candidate policy offline at several sizes in a single pass. If the
    hit rate on a few hundred MB is low, this branch dies cheaply and nothing was built.
- **Speculative gather** across the sampler's candidate set (prefetch rows for the top-k next
  tokens while sampling) - the only way to beat the t+1 unknowability, and probably pointless
  next to the above.

## Sequencing

1. **E008** first (`GGML_CUDA_DISABLE_GRAPHS=1`) - one env var, and it tells us whether the
   mid-graph CPU split is defeating graph capture, which is the mechanism underneath all of
   this. If tg is unchanged, capture was already off and the host path is confirmed as the
   stall.
2. **(a) as a measurement, not a patch** - warm the table's range externally and rerun tg. If
   tg moves, faults are the cost and the `WILLNEED` patch is justified; if not, faults are
   irrelevant and F1 dies while the sync/graph-split story stands. Note the VRAM row cache is
   the *cheaper* long-term fix if this comes back positive: a few hundred MB per card instead
   of 30 GB of RAM held warm.
3. **Index-log + offline hit-rate study** comparing LRU / SIEVE / S3-FIFO / ARC at several
   sizes. No inference change needed, and it settles the policy question with data.
4. **(b) for pp** if pp at long context still matters after the tg questions are settled -
   there the whole row set is known up front, so no cache policy is needed at all.

Note what this *doesn't* resolve: E006 showed tg ~12-44x off its floor in both split modes,
and PLE is one layer with 16 rows per token. Whether that much time can actually hide in 16
page faults per token is not yet established - it depends on fault cost, which depends on
system RAM and NVMe behaviour, both still unknown (E008b). If warming the file does nothing,
the suspect is the graph split and sync, not the bytes.
