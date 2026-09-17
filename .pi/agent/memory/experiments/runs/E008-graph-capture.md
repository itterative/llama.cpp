# E008 - HIP graph capture is active; it is worth ~7%, and it is not the problem

- date: 2026-09-17
- machine: bench-4x-r9700-32g
- tier: T2
- status: done
- parent: E006
- build: `c9a59ef73` (11009), user's fork, same as E005/E006
- variable: graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`), everything else held to E005's
  command line: `-lm none -sm tensor -fa 1 -lzm auto -ot per_layer_token_embd=CPU -d 40960`

## results

| test | E005 baseline | graphs off | delta |
|---|---|---|---|
| pp512  | 397.5 +/- 9.0 | 395.9 +/- 7.5 | -0.4% |
| pp4096 | 512.0 +/- 3.6 | 479.3 +/- 3.2 | -6.4% |
| pp8192 | 546.8 +/- 7.0 | 509.3 +/- 5.9 | -6.9% |
| tg128  | 28.20 +/- 1.02 | 26.20 +/- 0.22 | **-7.0%** |

Raw: `../results/user/results-no-cudagraphs.csv.log` (untracked `.log`).

## what this refutes

**My E006 hypothesis is dead**: I proposed that the CPU-placed n-gram table might split the
graph and thereby *prevent HIP graph capture*, degrading decode into eager per-node
submission. It does not. Capture is clearly happening, because turning graphs off costs real,
repeatable time. That is the fourth mechanism of mine this branch has eliminated (collective
latency, bandwidth, split-mode choice, graph capture) and the record of each is what makes the
remaining search narrower rather than circular.

More useful than the refutation is the **bound it puts on launch overhead**. Disabling graphs
moves tg from 35.5 to 38.2 ms/token, so the entire cost of submitting every kernel launch in a
decode step, ~48 layers of them, is **~2.7 ms**. Launch submission is therefore not the
bottleneck even in the worst case, and with graphs on it is nearly free.

## what this leaves

Tg is ~35.5 ms/token. Corrected floor (and note E005's depth section: `-d 40960` really does
fill the KV, so these measurements are at ~40 k depth, not shallow): ~3.3 GB active weights
(6 B params) + ~1.0 GB KV + ~0.2 GB fp32 GDN state = **~3.9 GB/token**, ~0.98 GB per card,
**~2 ms floor** at a conservative 500 GB/s/card. So the gap is **~18x**. Excluded so far, each
by a measurement rather than by argument:

| mechanism | excluded by |
|---|---|
| weight bandwidth | E006: layer/tensor ratio 0.88, predicted 0.25 |
| cross-GPU collectives | E006: layer split does ~none and is slower |
| launch submission / graph capture | E008: worth 2.7 ms/token in total |
| PLE fetch path (page faults) | arithmetic ceiling: ~28 KB and 16-23 page touches per token is 0.1-4 ms, not 33 (see `plans/ple-prefetch.md`) - unless the box is swapping |
| dense attention over 40 k | partially: it is ~6 GFLOP/token across the 12 full-attention layers, i.e. **~0.9 ms** at the observed effective throughput - real, but ~2.5% of the token, so it cannot be the missing 33 ms on decode |

That last row is worth keeping in view for the QSA thread: at 40 k depth attention is roughly
half the *FLOPs* of a prefill token, but since pp runs at <=3.5% of WMMA peak, removing half
the arithmetic need not remove half the time. Whether it does is not answerable from these
numbers - it is a depth-curve question, which is E013.

What remains has to be work that is **per decode step, serial, host-side, and not removable by
graph replay**. The obvious class is *per-step graph construction and allocation on the host*:
building the graph, reserving the cvec, and `ggml_backend_sched_alloc_graph` for a graph of
this size are all host work that happens *before* replay, so graphs being enabled does not
eliminate them - and the CPU split makes it worse by fragmenting one graph into several
subgraphs that each need their own buffers and a handoff between them.

## Next probes, both flag-only

1. **`GGML_SCHED_DEBUG_REALLOC=1`** - runtime `getenv` (`ggml/src/ggml-backend.cpp:1865`),
   needs no rebuild, and its in-source comment describes this exact pathology: "we are
   interested only in situations where the graph was reallocated even though its size remained
   the same", with a reference PR. If decode reallocates every step, that is the smoking gun.
2. **`LLAMA_GRAPH_REUSE_DISABLE=1`** (`src/llama-context.cpp:279`) - the inverse test. If
   disabling graph *reuse* costs far more than disabling graphs did, per-step rebuild is where
   the time is.
3. **Concurrency scaling** - `llama-batched-bench -m <model> -c 4096 -b 2048 -ub 512 -npp 512
   -ntg 128 -npl 1,2,4` plus the usual `-sm tensor -fa 1 -lm none -ot ...` (note: `llama-bench` has **no**
   `-np`; in batched-bench the sweep flag is `-npl`), or llama-server with two concurrent slots. If tg scales super-linearly with parallel sequences, the cost is per-step
   host work and not per-token GPU work, because one host step would be serving more tokens.
   This is the cleanest discriminator of the three and does not depend on the debug hooks
   working.

## Also untested: a third split mode

`-sm` accepts `none,layer,row,tensor`, and E006 compared only the two endpoints. **`row` splits
weights across GPUs but keeps KV on the main GPU**, which makes it the intermediate that
decomposes "tensor split" into weights-split plus KV-split - the thing E006's binary
comparison could not separate. Filed as E012.

## Note on the spreads

tg stddev was 1.02 with graphs on and 0.22 with them off, so replay is the *noisier* of the two
modes here. With n=3 that is barely evidence, but if a later experiment shows unstable tg,
"graphs on" is the first thing to suspect, not thermals.
