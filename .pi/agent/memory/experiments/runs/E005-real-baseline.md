# E005 - first real qwen4exp numbers (bench box, user's fork)

- date: 2026-09-17
- machine: bench-4x-r9700-32g
- tier: T2
- status: done (data intake)
- parent: E002 (which killed the T1 baseline route)
- build: `c9a59ef73` build 11009 - **user's fork, not this branch** (`ebbb18522` = 11024)
- model: `bartowski/Qwen3.8-Flash-Next-GGUF` **Q4_K_M**, file 119,588,006,400 B (119.6 GB),
  176.94 B params. GGUF `size_label` says **A3B**, but the user believes it is ~**6 B active**
  and is double-checking. Everything below is therefore computed for **both** cases; the
  conclusions do not flip (which is the useful property - the finding is robust to this input).
- flags, verbatim: `-lm none -sm tensor -fa 1 -lzm auto -ot per_layer_token_embd=CPU -d 40960 -p 512,4096,8192 -n 128 -r 3`

## results

| test | t/s | stddev | ms/token |
|---|---|---|---|
| pp512  | 397.5  | 8.99  | 2.52 |
| pp4096 | 512.0  | 3.59  | 1.95 |
| pp8192 | 546.8  | 6.98  | 1.83 |
| tg128  | 28.20  | 1.02  | **35.5** |

n = 3 each. Spread is 2-4%, so the measurement itself is well-behaved.

### Depth: corrected twice, and the second correction is the one that matters

I first read `-d 40960` as *allocation only* and wrote a caveat saying these were
shallow-context measurements. **That was wrong**, and the corrected reading is more
interesting: `llama-bench.cpp:2408-2433` really does fill the context -

```c
if (t.n_depth > 0) {
    bool res = test_prompt(ctx, t.n_depth, t.n_batch, t.n_threads);   // processes n_depth tokens
    cstate.depth = t.n_depth;                                        // state saved via llama_state_seq_get_data
```

and it caches that state across repetitions (`is_cached` at `:2409`, restored with
`llama_state_seq_set_data`), so the fill happens once and is **not** inside the timed region.

Consequences, all of which strengthen the baseline rather than weaken it:

- **every number in this record was measured at ~40 k context depth**, including `tg128`.
  That is the regime that actually matters for this model, not a toy one.
- the tg floor has to include KV. 12 full-attention layers x 2 KV heads x 256 x (K+V) x f16
  = ~24.6 KB of cache per token, so at 40960 depth a decode step reads **~1.0 GB of KV** on
  top of ~2.7 GB of active weights (plus ~0.2 GB of GDN recurrent state, fp32, 36 layers).
  Call it ~3.9 GB/token -> ~0.98 GB/card -> **~2 ms floor at a conservative 500 GB/s/card**.
- so the gap is **~18x**, not the 25-40x implied by the earlier weight-only floor. Smaller,
  and still far too large for bandwidth to be the story. E006's layer/tensor ratio of 0.88
  refutes bandwidth-bound independently of this arithmetic, so the conclusion stands on the
  measurement rather than on my estimate.
- one thing this *does* reopen: attention over 40 k was live in every measurement, and
  `-fa 1` was on throughout. If attention were ~2 ms of a 35.5 ms token, that is exactly the
  kind of contribution my "no KV in these numbers" error would have hidden. See E013, which
  survives but with a different justification: we have **one depth point, not a curve**.

## hardware and stack, from the same session

| field | value |
|---|---|
| GPUs | 4x Radeon AI PRO R9700 (Navi 48), **gfx1201** (0x1201), 32624 MiB each, 130496 MiB total |
| VMM / wave | `VMM: no`, `Wave Size: 32` on all four (same `GGML_HIP_NO_VMM` default as the dev box) |
| ROCm | **7.15.0** (dev box is 6.4.4) |
| OS | Fedora 44, kernel `7.2.5-200.fc44`, amd-smi `26.5.0`, VBIOS `00158738` |
| power | 210 W cap/card; idle 30-40 W; snapshot during prefill **92-95 W, 47-49 C** |
| topology | all four cards hang off the Zen3 (Starship/Matisse) root complex through two levels of PCIe switches; `BDF` 0b/10/13/19. **No XGMI/Infinity Fabric links** on Navi 48, so every inter-GPU transfer traverses the CPU |
| util snapshot | **GFX 9% on all four cards during prefill** |

Raw: **not in git, deliberately.** `../results/user/*.log` are untracked on purpose
(llama.cpp ignores `*.log` repo-wide, `.gitignore:17`), so they exist only on this disk.
What is committed is `../results/E005-summary.txt` - the transcription of record, holding
every number cited below plus the full flag set and hardware facts. `llama-bench.log` in the
raw set is empty (stderr went to `results.log`).

## validity - is this a baseline?

**No, and it should not be treated as one.** Rule 2 (never compare across commits or
machines) disqualifies it on three independent counts, and the user volunteered all of them:

1. the fork carries **RDNA4 MMQ fixes** that are not on this branch;
2. it carries a **custom AllReduce** (RCCL does not work on this setup) - and RCCL is also
   not compiled in here (`GGML_HIP_RCCL=OFF`), with the "rebuild with NCCL" warning
   suppressed on HIP, so the fallback on this branch is silent;
3. the user **enabled `-sm tensor` for qwen4exp in the fork**, and on this branch
   `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1161`) makes the load path *throw*
   `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'qwen4exp'`
   (`src/llama-model.cpp:358`). Guard added `d6f303004` (Apr), qwen4exp added to the
   not-supported list `36b101543` (Sep 1).

Plus a fourth the user did not mention: **ROCm 7.15.0 vs 6.4.4**, i.e. different backend
behaviour as well as different code.

So no number in the table above can be an A/B partner for anything measured on
`ebbb18522`. What it *is* good for is exactly as valuable, and it is why the run was worth
doing: it is a **ceiling demonstration and a calibration** - it proves a known-good
configuration on this hardware and says how far the current branch is from it.

## The actual finding: neither pp nor tg is bandwidth- or compute-bound

Active params: **~6 B** (resolved after this record was first written - the GGUF's `A3B`
`size_label` is parsed from the file name and is not to be trusted, backlog F2). At Q4_K_M
~0.55 B/param that is ~3.3 GB of weights per decoded token, and `-sm tensor` splits it across
the four cards.

- **tg**: 35.5 ms/token. Corrected floor, since `-d 40960` really does fill the KV (see the
  depth section above): ~3.3 GB weights + ~1.0 GB KV (12 full-attention layers x 24.6 KB per
  token-position at 40960 depth) + ~0.2 GB fp32 GDN state = **~3.9 GB/token**, so ~0.98 GB per
  card, ~2 ms at a conservative 500 GB/s/card. Measured is **~18x** that. E008 separately bounds
  all launch-submission cost at ~2.7 ms, and E006 refutes bandwidth by ratio (0.88 measured vs
  ~0.25 predicted), so the gap is not bandwidth and not launches - and it is not KV, since KV is
  already counted here.
- **pp**: at pp8192, 1.83 ms/token, ~12 GFLOP/token of active compute for 6 B params ->
  **~6.6 TFLOP/s aggregate**. RDNA4 WMMA fp16 peak is ~180-190 TFLOPS (user-reported; **per card
  or per box not yet confirmed**). Either way that is ~0.9% or ~3.5% of peak. Prefill is not
  compute-limited by any margin, which agrees with the 9% util snapshot - though that snapshot
  was taken during prefill, so it is the right half of the picture for pp and says nothing about
  decode.
- Corroborating oddity: **pp t/s rises monotonically with prompt size** (397 -> 512 -> 547).
  Attention work per token *grows* with context, so if attention were the bottleneck pp
  would fall. Rising means fixed per-ubatch costs dominate and are being amortised.

The machine is waiting, not working. That **inverts this branch's priorities**: cross-GPU
sync, graph splits and host-side stalls move to the front, and per-kernel work (the QSA
compaction port, H4b) moves back - a pp-only optimisation for a subsystem that is not the
bottleneck at 8k context.

### Candidate mechanism for tg - RETRACTED by E006

I proposed here that ~288 cross-GPU collectives per token (`-sm tensor`, 6 matmuls x 48
layers) at 50-150 us each over PCIe - with no Infinity Fabric on Navi 48 - "bracketed" the
measured 35.5 ms/token. **E006 refuted it by sign**: with `-sm layer` there are essentially
zero collectives, and layer split is *slower*, not faster. A mechanism that predicts the
wrong inequality is wrong however neatly its numbers fit, and the latency figure was too
generous to PCIe anyway. Kept here rather than deleted, because "a plausible story that fit
the arithmetic" is exactly the failure this ledger is meant to catch.

E006 also refuted bandwidth-bound (predicted layer/tensor ~0.25 if the cards are used
serially, measured 0.88). What survives is narrower and more useful: the tg cost is serial
and shared by both split modes, ~12-44x above the floor, and therefore lives in the host-side
path - see E006 and the E007/E008 proposals.

Caveat on the util figure: `GFX-Uti` is a one-shot sample, not a profile, and it was taken
during *prefill*, so it says nothing directly about decode. A proper per-kernel attribution
(`GGML_HIP_EXPORT_METRICS`, still unverified) is what turns "strongly suggests" into "shows".

### Derived: the lazy table is never prefetched, in *any* load mode

Reading `src/llama-model-loader.cpp` and `src/llama-mmap.cpp` properly, the situation is
different from (and worse than) the one I first wrote here, so the correction is part of the
record:

- lazy read applies to any tensor over 4 GiB regardless of name (`loader:1093`), so the ~30
  GB Q5 table qualifies;
- `-lm none` sets `use_mmap = false` (`:559`), yet `init_mappings()` still maps the file when
  `lazy.any()` (`:1411-1412`, with a comment saying lazy is deliberately usable without
  `--load-mode mmap`);
- `prefetch_size = prefetch && use_mmap ? -1 : 0` (`:1429`), so `-lm none` passes 0;
- **but even with `-lm mmap`, the lazy ranges are excluded from prefetch**: the WILLNEED loop
  iterates `ranges_complement(lazy_ranges, ...)` (`llama-mmap.cpp:500-502`), `MAP_POPULATE`
  is skipped because it "would fault in the lazy ranges too" (`:481`), and the lazy ranges are
  then marked `POSIX_MADV_RANDOM` (`:508-510`), which switches kernel read-ahead *off* for
  them.

So `-lm none` costs nothing extra for the table specifically: no mode warms it. The table is
demand-faulted with readahead disabled by design, which is the right choice for a hash gather
*within* a page-resident range and the wrong one when the range is not resident at all.

What is actually missing is a way to say "cold once, then hot": an explicit
`POSIX_MADV_WILLNEED` over the lazy ranges (or `-lzm` variant that pre-warms them in a
background thread while the GPU tensors upload). ~30 GB of page cache would then absorb the
gathers. That is backlog F1, and it is a sharper question than "does prefetch apply when
`-lm none`", which it does not.

Relevant to whether this even matters: `MADV_RANDOM`/prefetch behaviour is also suppressed on
NUMA (`if (numa) { prefetch = 0; }`, `:473`, plus a whole-file `MADV_RANDOM` at `:516-522`),
so the box's NUMA topology belongs in the answer - and system RAM is still unknown. Note also
that `MADV_RANDOM` per gather means ~one fault per 2560-row read: with `ple_n_heads = 16`,
that is up to 16 minor-or-major faults **per token**, which for `tg` at 35.5 ms/token is
potentially visible if any of them are major.

## open questions for the user

1. What is the exact fork delta? Specifically the MMQ fix, the custom AllReduce, and the
   `sm_tensor` enablement (which is the one most likely to bite when you merge forward -
   on current master your command line throws before loading).
2. System RAM, and does the box have NVMe swap/page-cache pressure at steady state? That
   decides whether L1/E006 can matter at all.
3. Does the server-vs-bench pp difference appear with the *same* flags? The user reports
   they match at low context and the server pulls ahead (~1k t/s) at depth. Since attention
   cost grows with depth, a rising curve suggests either amortisation of fixed cost or a
   state difference between the two paths (`-lm none` here vs server's default mmap, KV/prefix
   reuse, concurrent slots, ubatch accumulation). Worth pinning down before any pp A/B -
   otherwise we would be optimising whichever instrument we happened to use.
