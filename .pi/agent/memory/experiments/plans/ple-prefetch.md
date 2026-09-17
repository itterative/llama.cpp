# PLE / n-gram table: prefetch design notes

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
