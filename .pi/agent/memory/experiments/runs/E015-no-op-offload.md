# E015 - op offload makes no difference (negative)

- date: 2026-09-17 | machine: bench-4x-r9700-32g | tier: T2 | status: done | parent: E011
- build `c9a59ef73`, E005 config plus `-nopo 1` (`--no-op-offload`)
- raw: `../results/user/results-nopo.log` (untracked `.log`)

| test | base (E005) | `-nopo 1` |
|---|---|---|
| pp8192 | 546.83 +/- 6.98 | 547.50 |
| tg128  | 28.20 +/- 1.02 | 28.14 +/- 1.01 |

Zero effect, and unlike E010 this one is verifiable from the data rather than inferred from the
effect: the CSV records `no_op_offload=1`, so the setting demonstrably applied.

The hypothesis was that the scheduler offloading some ops to CPU per step was the fixed cost.
It is not. Note the distinction from E011's `-ot` finding: `-nopo` governs *scheduler-chosen*
op offload, whereas the PLE table's placement comes from lazy read forcing a CPU buffer type,
which `-nopo` does not touch. So this result does not clear the CPU-placed table - only the
scheduler's own offload decisions.
