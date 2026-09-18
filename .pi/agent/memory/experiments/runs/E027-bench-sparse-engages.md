# E027 - sparse FA now shows on the bench; tg credit is confounded

- date: 2026-09-18 | machine: bench-4x-r9700-32g (user's fork) | tier: T2 | status: open
- source: `results/user/results-qsa-experiments.log`, same command as E025 (`-sm tensor -fa 1
  -lzm auto -ot per_layer_token_embd=CPU -d 4096,16384,40960,131072 -p 512,4096,8192 -n 128 -r 3 -b 2048 -ub 1024`)
- arms: baseline `c9a59ef73` (11009) / qsa-A `8283df294` (11011) / qsa-B:OLD `5480fa2c7` (11013,
  superseded by E025's merge problem) / **qsa-B `92a47455f` (11062)** with `Q4EXP_SPARSE_FA=1`

## what these arms can and cannot say (user correction, mine to make)

The baseline and qsa-A runs were taken **before the rebase and the conflict fixes**, so three of
the four arms are stale-base and there is currently **no control at the new base at all**. What
survives:

- **A's effect is internally valid but measured at the old base.** `c9a59ef73` vs `8283df294`
  differ only by my two host commits in a file that never conflicted, so +4.85% tg at 131k,
  +1.5% at 40k and ~0 below is a real A-vs-noA reading - of that base.
- **newB's pp shape proves engagement**, because the depth profile matches the gate exactly
  (~0 at 4096, rising to +21.6% at 131k). That is qualitative evidence and does not need a control.
- **Nothing in the "newB vs qsa-A" column is an effect size.** It mixes bases (11011 vs 11062) and
  includes the user's p2p allreduce and mmq landing, so both its pp and tg numbers are unattributed.

What is actually missing, at `92a47455f` or later: **the same build with the flag unset** (splits
sparse from their landing work), and a build with `3aea8533a`+`798f54bd9` reverted if A's number is
wanted at the new base. One build plus one env toggle covers the first, which is the one I would
prioritise.

## qsa-A at the old base

Same two builds as E025: +4.85% tg at 131k, +1.53% at 40k, <=0.5% elsewhere, no pp effect. Valid
as an A-vs-noA comparison, but only for that base.

## sparse FA engages, with the right signature

newB against the stale-base qsa-A numbers (magnitudes unattributed, shape valid):

| depth | pp512 | pp4096 | pp8192 | tg128 |
|---|---|---|---|---|
| 4096   | +0.78% | +0.60% | +0.68% | +6.17% |
| 16384  | +2.51% | +0.94% | +0.94% | +5.91% |
| 40960  | +4.66% | +3.06% | +1.95% | +6.05% |
| 131072 | **+21.59%** | **+10.36%** | **+12.41%** | +3.61% |

The pp shape is what sparse attention should do: nearly nothing at 4096 (below the `K->ne[1] >=
max(4096, 2*n_kv_max)` gate), then rising with depth to +21% at 131k. On the dev box the same
mechanism gives +21% at 40k and +55% at 164k, so the bench is responding less at equal depth,
which fits four cards already being further from bandwidth-bound on attention. The OLD arm's
flatness is now explained: that build had the conflicts resolved badly, and it is marked as such
in the log.

## why the tg column must not be credited to sparse FA either

The two binaries are 51 commits apart: qsa-A is `11011`, qsa-B is `11062`, so the newer one also
carries the user's `wip: p2p allreduce` and the mmq/quant retune landing. Three independent reasons
to expect the +6% tg to be theirs, not mine:

1. **Decode does not use the kernel that has the sparse path.** P4 established that `mma_f16`
   serves prompt processing while 1-token decode falls to tile/vec, and `may_use_sparse` lives only
   in `mma_f16` - so sparse cannot engage in tg at all.
2. Locally (E020, E026) tg was flat under the flag at every depth, with the same code.
3. A uniform +6% across all four depths, including 4096 where the pp gate is not even passed, is
   the shape of a collective/host-path change, not of attention traffic.

The clean control is one run: **same build `92a47455f`, flag unset.** That splits their intervening
work from my change for both pp and tg, and it is the number worth having.

## consequence for E011

If `92a47455f` without the flag already gives ~+6% tg, then the bench's fixed per-step cost has
moved since the 28 ms measurement, and every E011-era percentage on this branch should be re-read
against a fresh baseline. That is a bigger deal than the sparse result.
