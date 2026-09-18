# E026 - re-baseline after the user's rebase, and qsa-A/qsa-B re-checked on top of it

- date: 2026-09-18 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- tree: `a8b24dfdf` = my four code commits (all still ancestors, markers verified) + the user's
  seven: q2_k/q6_k mma fixes, RDNA3/4 mmq config retune, quant sweep, p2p allreduce, `-sm tensor`
  re-enable
- context: the user briefly switched this worktree to `local/qwen4exp` to push, which read as the
  branch being wiped; it was a checkout, nothing was lost. One number taken during that window
  (tg 176.19 / pp 5928 at d40960) belongs to `local/qwen4exp`, not to this lineage.

## the fingerprint moved, and the cause is their quant work

| arm | before rebase | after rebase | delta |
|---|---|---|---|
| golden corpus, `-np 1`   | 262938.7619 | **263113.6984** | +0.067% |
| sparse corpus `-c 8192` dense  | 267157.4202 | 267035.3653 | -0.046% |
| sparse corpus `-c 8192` sparse | 267157.2589 | 267035.3524 | -0.046% |

`mmq-config-rdna4.cuh` changed by 558 lines plus `mmq.cu` and `mmq-vec-dot.cuh`, i.e. which
dequant/mma path serves each quant type. The dummy carries the real file's per-tensor types
(Q4_K/Q5_K/Q6_K/Q5_0/Q8_0), so a numerics shift is the expected outcome, not a regression: dense
and sparse moved by the *same* 0.046%, and dense-vs-sparse agreement is still ~5e-8 relative.

**New gate values** (E018's contract is "every diff explained", and this one is): golden
`263113.6984 +/- 3043.13362`; deep corpus at `-c 8192` dense `267035.3653`, sparse
`267035.3524`. The old values stay valid for anything built before `a8b24dfdf`.

## qsa-A and qsa-B still do what they did

| metric | my E023 value | this build | note |
|---|---|---|---|
| tg128 @ d40960  | 199.57 +/- 0.80 | **199.81 +/- 1.17** | host fix intact |
| tg128 @ d163840 | 125.19 +/- 1.22 | **124.89 +/- 1.23** | intact |
| pp512 @ d40960, sparse on vs off | 7467 vs 6054 (+23.3%) | **6556 vs 5407 (+21.2%)** | sparse still engages |
| pp512 @ d163840, sparse on vs off | 4013 vs 2533 (+58.5%) | **3772 vs 2431 (+55.2%)** | intact |

So both changes survive the rebase with the same relative behaviour, including the depth gate
(4102 KV) and the pp-only signature.

## the part that is theirs, and it may matter

The same tuning that legitimately moved the fingerprint costs prompt processing on this model:

| dense pp512 | before rebase | after rebase | delta |
|---|---|---|---|
| d40960  | 6089 | **5407** | **-11.2%** |
| d163840 | 2552 | 2431 | -4.8% |
| sparse arm d40960  | 7467 | 6556 | -12.2% |
| sparse arm d163840 | 4013 | 3772 | -6.0% |

Consistent in both arms and at both depths, decode untouched (0.1%), so it is not related to
qsa-A/B: it is the quant/mmq config path. Two caveats before reading it as a regression: the dummy
weights pp toward MoE expert matmuls (512 experts, the real per-layer shapes) so it may be
oversampling exactly the configs that were retuned for other shapes or other quant mixes, and the
real file's distribution of types across 48 layers is not identical to mine in weight. But the
depth dependence is the right shape for an mmq effect (-11% at 40k where mmq dominates, -5% at
164k where attention takes a bigger share), so it is worth checking against their bench numbers
rather than dismissing.

## bookkeeping

The backlog had reserved E025/E026 as run ids; E025 became the bench validation and E026 this
re-baseline, so the memoization idea moved to E027.
