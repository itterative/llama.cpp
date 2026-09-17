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

Active params: **3 B or 6 B, unresolved** (see above). At Q4_K_M ~0.55 B/param a decoded
token touches ~1.65 GB or ~3.3 GB of weights; `-sm tensor` splits that across four cards.

- **tg**: 35.5 ms/token. At a conservative 500 GB/s/card, the per-card weight read is ~0.41 GB
  (3 B case) -> ~0.8 ms, or ~0.83 GB (6 B case) -> ~1.7 ms. Measured is **21-44x** that. Even
  allowance for KV reads, the 16 PLE gathers and 48 layers of launch overhead cannot cover a
  gap that size, and neither case makes it bandwidth.
- **pp**: at pp8192, 1.83 ms/token. Compute is ~6 GFLOP/token (3 B) or ~12 GFLOP/token (6 B),
  so 546.8 t/s is **3.3 or 6.6 TFLOP/s aggregate**. RDNA4 WMMA fp16 peak is ~180-190 TFLOPS
  (user-reported; **per card or per box not yet confirmed**). Every combination lands between
  ~0.4% and ~3.5% of peak. Prefill is not compute-limited by any margin, which agrees with the
  9% util snapshot.
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
