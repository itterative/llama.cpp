# Plan: decode communication cost on 4x R9700 (`-sm tensor`)

Supersedes the motivation and the host-flow section of
[p2p-legacy-improvement-plan.md](p2p-legacy-improvement-plan.md) (the push-based one-shot
sketch, written without multi-GPU access). That design's algorithm, memory layout and PCIe
ordering argument all still stand and are referenced below rather than repeated. What changed
is what we are optimizing for, because the box has now been measured.

Evidence: `results/user/llama-bench/82bc067c3/0[3-6]*.log`, `99-*.log`, `rccl_debug_verbose.log`,
plus E053 (131k decode-window kernel stats) and H16.

## 1. Measured, 2026-09-25

All arms: real Qwen3.8-Flash-Next Q4_K_M, 4x R9700, `-sm tensor -fa 1 -d 4096,131072 -p 0 -n 128
-r 3`, `Q4EXP_POOLED=1 Q4EXP_SPARSE_FA=0 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 GGML_PROF_REGIONS=1`,
build `ea2c69a30`. `us/call` is the host-side `meta:allreduce` region, per collective.

**Phase caveat on the `us/call` column.** Those values were isolated by subtracting the first
per-test region report from the exit report, and the 11746 calls that came out are `128 ubatches x
96`, i.e. the d131072 *fill*, not the decode reps (which would be 384 steps x 96 = 36864). So the
absolute per-call costs are fill-phase and the `ms/step` column is really per fill ubatch. The
cross-arm *ordering* is still sound - every arm ran an identical phase mix, and a collective's API
call count does not depend on which phase issued it - but the pass-through estimate below compares
mixed-phase host cost against decode-only tg, so treat it as soft until P0 replaces it.

| arm | us/call | ms/step | tg d4096 | tg d131072 |
|---|---|---|---|---|
| `05` `NCCL_ALGO=Ring` (the default) | 37.0 | 3.55 | **33.48** | **29.61** |
| `04` `NCCL_ALGO=Tree` | 37.6 | 3.61 | 32.92 | 29.00 |
| `99` `RCCL_USE_AMD_SMI_LIB=1 NCCL_CUMEM_ENABLE=1` | 40.1 | 3.84 | 3.60 | 3.40 |
| `06` `GGML_CUDA_ALLREDUCE=internal` (our p2p, auto -> butterfly) | 127.1 | 12.20 | 31.23 | 28.05 |
| `03` `NCCL_ALGO=FC` | 133.4 | 12.81 | 30.78 | 27.45 |

## 2. Cost model that fits these numbers

**Host cost per collective is linear in the number of HIP/NCCL API calls, at roughly 4-5 us per
call on this stack.** The NCCL path enqueues `ncclGroupStart` + one `ncclAllReduce` per rank +
`ncclGroupEnd` = 6 calls (`ggml-cuda.cu:1019-1032`) -> 37 us. The copy-based butterfly issues,
per device per round, a peer copy, an add kernel or a D2D copy, an `cudaEventRecord` and a
`cudaStreamWaitEvent`, and records all `n` events each round (`allreduce-p2p.cu:809`, `:835-841`,
`:908-925`) -> ~28-32 calls -> 127 us. 6 -> 37 and ~30 -> 127 is the same line.

**Host time does not pass through to wall 1:1 - roughly a quarter, soft.** `06` and `03` carry ~9
ms/ubatch more host time than `05` and lose 2.25 and 2.70 t/s of decode, i.e. ~2.1-2.6 ms/step. So
host work is not free, but a ms of it is worth well under a ms of step time; the rest overlaps
device work. P0 replaces this with a per-phase measurement.

**The ~4-5 us constant is now measured independently, and it is 6.3 us.** `tools/peer-probe.cu`
test5 does 200 sequential single-thread remote stores: 6.33 us per launch+store. NCCL's 37 us over 6
calls is 6.2 us per call. That is the model confirmed from two directions, and it sets a floor: at
N=4 a one-shot collective is 4 launches, so ~25 us of host time against NCCL's 37. **The host-side win
is therefore ~1.5x, about 1% of tg, not the 4x the call-count arithmetic suggested** - which moves the
justification entirely onto the device side and makes P3 (AR inside the captured graph, where per-AR
host cost stops existing) the real host lever.

**RCCL is exhausted as a lever.** `04` shows Tree was silently ignored (identical per-call cost to
Ring), consistent with its `AllReduce | Tree = 0.0/0.0` row in the tuning table. `03` shows FC is
real and worse. LL and one channel are already auto-selected at 10240 B. `VMM: no` on all four
devices means `cuMem`/symmetric windows cannot exist here, so no NCCL one-shot path is available
at any setting. `99` is a hazard, not a data point: that pair loses 9x while making the collective
cheaper, and the loss is outside every region we measure.

## 3. What this falsifies from the legacy plan

- *"~200 us fixed per AR, independent of API-call count, pointing at the event/handshake machinery
  itself."* NCCL uses no app-visible events and no handshake machinery and still costs 37 us host
  and ~60 us device per call. Cost scales with API calls, and our event-heavy path is only 127 us,
  not 200+. The fixed-cost theory does not survive contact with the box.
- *Sim ordering (butterfly 48.6 vs ring 41.3 t/s).* Measured: ring 33.48 vs butterfly 31.23. The
  sim's host model has the two paths the wrong way round, so its projections (87-115 t/s) should
  not be used to justify effort. Re-derive from the measured per-call costs instead.
- *"8 kernel launches per AR at N=4, zero event API calls"* as the target host flow. At ~4.5 us per
  call that is ~36 us, which **ties NCCL**. The legacy plan splits the work into a push kernel and
  a reduce kernel per device; two launches per device is not an improvement. One-shot is only worth
  building if it is **one fused kernel per device per collective**.

## 4. Design consequence: fuse push and reduce into one kernel

Per device, one launch, in program order: write my tensor into each peer's inbox slot (grid.y over
peers, uint4), `__threadfence_system()`, bump a local block counter, last block rings the doorbell
on all peers (plain volatile store, posted write, ordered after data by the fence), spin on *local*
flags until every peer's generation lands, `__threadfence()` acquire, then sum own tensor plus inbox
slots in ascending device order into the output. The legacy plan's layout, flag semantics and
"no remote atomics anywhere except the local counter" carry over unchanged.

Two additions the sketch does not have, both load-bearing:

- **The grid must be guaranteed co-resident.** Blocks that have not launched cannot contribute to
  the doorbell, and blocks that spin in the wait phase hold their SMs, so a grid larger than
  resident capacity deadlocks. Decode payload is 10 KB, so a fixed small grid (e.g. 32 blocks x
  512 threads on 54 CUs) is comfortably enough. State the bound in code and assert against it.
- **Fold the inactive-shard zeroing into the push kernel.** Today a tensor without
  `GGML_TENSOR_FLAG_COMPUTE` is `cudaMemsetAsync`'d first (`ggml-cuda.cu:1021-1027`); in the fused
  kernel the push for a non-contributing shard just writes zeros. Saves a call and keeps NCCL
  semantics.

Host flow per collective becomes `n` launches (4 here) versus NCCL's 6 calls: ~16-20 us. Below the
measured baseline, which is the bar.

## 5. Where the prize actually is, and what decides it

The host-side prize is small and known: 37 -> ~18 us/call is 1.8 ms/step, times 25% pass-through
is about **+1.5% tg**. Do not build a kernel for that.

The device-side prize is the reason to consider it: E053 measured the NCCL collective at 60.7 us/call
median with **min 57.7**, i.e. a tight floor rather than a skew-shaped distribution. 96 calls/step
puts ~5.8 ms/step of device-resident time inside the allreduce chain, and that time is on the
dependency path by construction (the next node needs the sum). A 4-rank ring is 6 dependent PCIe
half-steps; a one-shot is one concurrent push plus one concurrent read. If the floor is hop count
times hop latency, the device cost should fall toward 15-20 us/call, which is ~4 ms/step - roughly
**+13% tg** if it passes through anywhere near 1:1.

Whether device time passes through is *unknown*, and E053's other finding argues it may not: about
half the step has no kernel resident at all, which is a host-bound symptom. So:

**Step 0, before any kernel, ~30 minutes, no code changes:**
- Run `llama-bench` at a *single* depth each (`-d 4096` and `-d 131072` as separate invocations).
  The sweep above mixed each depth's fill with its decode inside cumulative region reports, so
  `meta:subgraph` and `graph:set_inputs` cannot be attributed to the decode step at all. Single
  depth makes the exit report approximately decode-only and gives the real per-step host budget.
- Trace arm `05` with `GGML_PROF_DECODE=1` (rocprofv3, whole `-o` dir) and read the NCCL kernel's
  min/median duration and the busy-sum against `phase:decode` wall.
- **Anchor on the last per-test region report, not the first.** The sweep's per-call numbers above are
  fill-phase because of exactly this mistake.
- Go/no-go: build the one-shot only if (a) the NCCL kernel's min duration is a floor rather than a
  spread, and (b) busy-sum/wall says the device is on the critical path. If instead host gaps
  dominate, the whole effort redirects to P3.
- Read `graph:alloc` and `sched:realloc_size` in the same single-depth logs. In the mixed deltas above
  they came out as ~123 calls at ~420 ms each, about 52 s of one run, i.e. one re-reserve per
  1024-token fill ubatch. That may be the same phase-attribution artifact, but if it is real it
  dwarfs every comms number in this plan and contradicts E051's "one re-reserve left". A `-d 131072`
  run reading the re-reserve count and total settles it.

## 6. Milestones

1. **P0 - measure.** Above. Also add a temporary counter of actual API calls per collective in both
   paths under an env flag, ~10 lines, to confirm the 4-5 us/call line instead of inferring it.
2. **P1 - doorbell ordering torture, before anything else.** Standalone harness, 2 GPUs, 1M
   push/flag cycles at varying sizes, verify data-vs-flag ordering every iteration. The whole design
   rests on posted-write ordering plus a system fence on this root complex; NCCL-LL and TRT-LLM make
   the same assumption but usually on NVLink. The legacy plan's risk item 1, unchanged, and it is
   still first. Note the legacy plan calibrated on the dev box's Ryzen 5800X - re-check the bench
   box's host CPU before trusting that discussion.
3. **P2 - fused one-shot kernel**, 1 launch/device/collective, `GGML_CUDA_AR_DIRECT_ALGO=oneshot`,
   opt-in via `GGML_CUDA_AR_ONESHOT=1`, `auto` keeps today's behaviour. Accept: us/call <= 20, tg
   d4096 and d131072 both >= NCCL arm, cross-device results bit-identical, golden unchanged
   (`-sm tensor` reference 263113.7846).
4. **P3 - attack the host gaps** if P0 says they dominate: get the collective *inside* each device's
   captured graph (`use_cuda_graph` is already true, `ggml-cuda.cu:2631`, and the AR currently sits
   between subgraphs in host code). Requires the generation counter to live in device memory rather
   than as a kernel argument, because a captured arg is a constant and would repeat forever. This is
   the only path that removes ~3.5 ms/step of host work outright rather than halving it.
5. **P4 - fuse the consumer with the collective.** The endpoint shape: an AR that writes its result
   straight into the residual add and following norm deletes nodes instead of speeding one up, and
   deletes the launch that would have waited on it. Bigger than P2 in wall-clock terms and a graph
   change, not a kernel change. Park until P2 measures.
6. **P5 - policy.** Crossover `GGML_CUDA_AR_DIRECT_ONESHOT_BYTES` from data (legacy plan guessed
   256 KiB); keep prefill on the existing path, since 1024 x 2560 bf16 = 5 MiB collectives are
   bandwidth-bound and ran `proto SIMPLE` in `rccl_debug.log`.

## 7. Non-goals

- No more RCCL env work. Ring+LL is its best and it is already selected; Tree and FC are measured
  dead ends and `99` is a hazard, not a knob.
- No `GGML_CUDA_AR_DIRECT_RING_ORDER` effort: RCCL built `0 1 2 3` and the topology table has all
  twelve pairs at identical weight and 2 hops, so there is no better permutation to find.
- Do not iterate on the copy-based butterfly or bde to make one-shot-shaped. Halving the rounds in a
  ~30-call path still lands at ~64 us/call, i.e. above NCCL.
- `-sm layer` is off the table (user decision).

## 8. Implementation status (2026-09-25)

P2 landed in `ggml/src/ggml-cuda/allreduce-p2p.cu`, opt-in, default untouched.

**One deliberate deviation from the legacy sketch, and it removes P1.** The wire unit is
NCCL-LL-shaped rather than doorbell-shaped: each 8-byte store carries 4 bytes of payload plus a
4-byte generation tag in the *same* atomic word. A reader therefore sees either the old unit or the
new one, so there is no flag array, no arrival counter, no `__threadfence_system()` anywhere, and no
separate wait phase - the read loop is the wait. **That deletes the plan's top risk item**: the
posted-write-ordering assumption never comes up, because there is no flag that could overtake data.
Inboxes are double-buffered by generation parity (proof sketch in the code comment: a peer's g+2 push
is enqueued after its g+1 kernel, which needed my g+1 push, which is after my g kernel finished).

What is in: `GGML_CUDA_AR_DIRECT_ALGO=oneshot`, one kernel per device per collective, inactive shards
zeroed inside the push instead of a `cudaMemsetAsync`, no `data_ready` events recorded on this path,
`GGML_CUDA_AR_ONESHOT_PROBE=<iters>` running the real path over the real links at init and dropping
to `auto` if the sum is wrong, and a co-resident 16 x 256 grid.

What was **not** verifiable locally, because this box has one GPU: peer-pointer dereference across
the root complex, that a remote posted write becomes visible to a local volatile load at all, the
spin's forward progress, and any timing. The probe is the first thing to run for exactly that
reason. The spin has a 2e8-iteration bound that prints and contributes zero - chosen deliberately
over hanging the box, which means a failure looks like a loud log line plus wrong numerics rather
than a wedge.

Two bugs found by re-reading rather than by testing, both would have been box-only failures: the
inbox was `cudaMalloc`'d uninitialized so a random tag could equal generation 1 (now zeroed, tag 0
means never-written), and the spin load was not `volatile` so the compiler could hoist it out of the
wait loop (would have hung or read stale data).

### Test recipe on the bench box

```sh
git pull   # file is existing, so no cmake -B needed, but reconfigure is harmless
export LD_LIBRARY_PATH=$PWD/build/bin

# 1. probe only: does the LL unit survive the root complex, and what is one round trip worth
GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_ALGO=oneshot \
GGML_CUDA_AR_ONESHOT_PROBE=200 \
  build/bin/llama-cli -m <model> -p hi -n 1 -ngl 99 -sm tensor 2>&1 | grep -i "ar-oneshot\|probe"

# 2. correctness: golden must be unchanged vs the -sm tensor reference 263113.7846
GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_ALGO=oneshot \
  build/bin/llama-perplexity -m <model> -ngl 99 -lm none -sm tensor -fa 1 -f <golden-corpus> 2>&1 | grep "Final estimate"

# 3. perf: same arms as the five-arm sweep, so us/call is directly comparable to NCCL's 37.0
GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_ALGO=oneshot GGML_PROF_REGIONS=1 \
Q4EXP_POOLED=1 GGML_FATTN_RDNA_RTILE=1 GGML_CUDA_MMVQ_RDNA4_SMALL_K=1 Q4EXP_SPARSE_FA=0 \
  build/bin/llama-bench -m <model> -lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU \
  -d 4096,131072 -p 0 -n 128 -r 3 -b 2048 -ub 1024 -o md
```

### Results

Three arms on the box at build `e9769ef02`, all with `GGML_PROF_REGIONS=1`, dense + pool + small_k,
`-d 4096,131072 -p 8192 -n 128 -r 3` (`results/user/p2p-improvements/run6-*.log`):

| arm | pp8192 d4096 | pp8192 d131072 | tg128 d4096 | tg128 d131072 | AR us/call |
|---|---|---|---|---|---|
| NCCL (platform default) | 1886.7 +/- 83.9 | 1230.9 +/- 8.1 | 33.10 +/- 2.00 | 29.76 +/- 1.59 | 15.2 |
| internal + auto (butterfly/bde) | 1787.3 +/- 95.0 | 1189.4 +/- 10.0 | 31.90 +/- 1.81 | 29.07 +/- 1.51 | 41.2 |
| internal + one-shot | 1837.9 +/- 85.8 | 1213.3 +/- 17.3 | **35.50 +/- 2.27** | **32.17 +/- 1.90** | **5.3** |

**Decode: +7.2% and +8.1% over NCCL, +11.3% and +10.7% over the existing internal path.** Best decode
numbers this box has produced for this model. Roughly two standard errors on its own, so the mechanism
carries the rest of the weight: host cost 15.2 -> 5.3 us/call is 2.9x, and the ~30% pass-through slope
predicts about the gain measured.

**Prefill is at parity, and the middle arm is what proves the cutoff works.** Prefill collectives are
~10 MB, so arms `b` and `c` execute the identical `bde`/`butterfly` path there - and still differ by
+2.8% and +2.0%. That difference is caused by nothing, which puts the pp noise floor of this harness at
about 3%, and puts `c` vs `a` (-2.6%, -1.4%) inside it.

**Two predictions of mine were wrong, both in the same direction: I under-priced how cheap a launch is
and over-priced the win.** I claimed a hard host floor of `n_devices x 6.3 us ~ 25 us` from
`peer-probe` test5; actual is 5.3 us, i.e. ~1.3 us per launch. test5 measured launch *plus a fenced
remote store plus a device sync*, not enqueue cost. The first repricing said the feature was worth ~1%
of tg; it is worth ~7%.

**The internal copy/event pipeline is worse than NCCL** (-3.6% tg), which retroactively justifies this
branch keeping NCCL as the default and means "our p2p is slightly slower" (the old H16 note) was
underselling it.

### What it took to get here

Three defects, none of which was visible on a single-GPU box, in order of discovery:

1. **`os_inbox[]` was always NULL.** Pipeline init allocates `dev_tmp` directly with `tmp_bytes`
   defaulting to 16 MiB, so `ensure_tmp`'s growth path - the only place I allocated the inboxes -
   early-returns forever. The kernel wrote to `NULL + slot offset`, which is the `0x2000000` page fault
   in dmesg, and the three peers then spun forever on data that would never come: `failed to suspend
   all gangs` on three cards, 4 s apart, with no fault recorded on them. Diagnosed by dumping the
   kernel's four pointers with `hipPointerGetAttributes` under `GGML_CUDA_AR_ONESHOT_DEBUG=1`.
2. **No fence after the push.** Atomicity of the 8-byte tagged store protects against torn reads, not
   against the write never leaving the source device. `peer-probe` test1 vs test2 shows an unfenced
   remote store is invisible to a kernel on the destination. Load-bearing, but *not* the cause of the
   hangs - the NULL deref explained those too, and I mis-attributed the multi-card pattern to it.
3. **Two log traps.** The probe result went to `GGML_LOG_INFO`, which llama-bench mutes without `-v`,
   so "no output" was about the sink. And the probe's timing loop synced all four devices per iteration,
   printing 153 us for a collective that costs 5.3 us; it now reports pipelined and fully-synced
   separately.

Also worth keeping: **peer stores surface to kernels on the destination, but a `hipMemcpy` of the same
address reads stale** (copy engine sees DRAM, peer writes land in the destination's cache). That is
harmless here because nothing DMA-reads an inbox, and it made the first version of `peer-probe` report
a false negative.

### Status and what is left

Opt-in: `GGML_CUDA_P2P=1 GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_DIRECT_ALGO=oneshot`. Correctness
evidence so far is a token-level match against the NCCL arm on real weights plus the init probe
(sum identical on all four devices) - not a PPL comparison, and not bit-exactness, which the different
reduction order rules out by design.

Open: the 256 KiB cutoff is a guess (the plan's P5), the `auto` ladder does not include one-shot yet, so
three env vars are needed to reach this path at all, `GGML_CUDA_AR_DIRECT_TMP_BYTES` defaults to 16 MiB
so the inboxes reserve 4 x 16 MiB per GPU, and whether `GGML_CUDA_ALLREDUCE` itself should stop
defaulting to NCCL on this branch is undecided.

## 9. Open questions

- Does P2's device win survive if the four ranks are only loosely synchronized by the ARs
  themselves? A spin kernel that keeps SMs busy while peers catch up is fine at decode batch 1 and
  wrong if the AR ever overlaps compute on-device.
- Is the 5 MiB prefill collective worth a two-shot variant, or leave it on NCCL permanently?
- `graph:alloc` / `sched:realloc_size` show ~123 calls at ~420 ms in these runs, which is H9
  territory and dwarfs every comms number here. It is very possibly the largest single host-side
  cost in a llama-bench decode measurement, and it was excluded from the sweep by design. Worth one
  dedicated look after P0, because a 420 ms reserve per fill ubatch is not what E051 claimed was
  left.
