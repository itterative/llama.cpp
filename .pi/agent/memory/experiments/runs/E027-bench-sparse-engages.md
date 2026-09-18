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


# CORRECTION - the control refutes this record's headline (same day)

The user redid qsa-A at the new base, which is the control this record said was missing. Same build
`92a47455f` (11062), flag off versus flag set:

| test | flag off | flag on | delta |
|---|---|---|---|
| pp512 @ d4096   | 408.96 | 408.47 | -0.12% |
| pp512 @ d40960  | 481.37 | 481.91 | +0.11% |
| pp512 @ d131072 | 476.17 | 476.54 | +0.08% |
| pp8192 @ d131072| 850.87 | 847.95 | -0.34% |
| tg128 @ d4096   | 34.73  | 35.30  | +1.64% |
| tg128 @ d131072 | 21.31  | 21.50  | +0.89% |

**Sparse FA does nothing measurable on the bench.** Every pp row is inside +-0.34%, and the tg rows
are inside their own noise and are not something the sparse path can even affect (decode does not
reach `mma_f16`). The +21.5% at 131k that this record attributed to sparse is in the *base drift*
column instead: qsa-A at 11011 versus the same code at 11062 gives +21.5% pp512@131k, +10-13%
pp4096/8192@131k, +2-4.5% at the shallower depths and **+4.5% tg at 4096**, where there is no depth
effect at all. So the whole of it was the user's landing commits, not my change.

Two candidate readings of that base drift, and the probe decides between them:

1. Their p2p allreduce and mmq retune simply made everything faster, with the long-context pp rows
   gaining most for reasons of their own.
2. **Their fork already engages sparse attention**, in which case 11062-with-flag-off and
   11062-with-flag-on are the same computation, which is exactly what the +-0.3% control shows. This
   would also explain why the gain is depth-shaped.

`tools/qsa-fa-probe.patch` (verified against this tree, `Q4EXP_FA_DEBUG=1`) prints the gate verdict
and every input it tested per FA op build. One short run with the flag **off** answers it: if the
gate prints TRUE without the env, their fork is doing sparse already and my ggml change is
redundant there; if it prints FALSE, the printed `n_kv_max`, `mask` and `K` shapes say which
condition rejects on the real model.

What still stands from this record: qsa-A's own effect (+4.85% tg at 131k against a build differing
only in a non-conflicting file), and that the bench's fixed per-step cost has moved - the +4.5% tg at
4 k depth in the base drift column means E011-era percentages need re-reading against a new baseline.
