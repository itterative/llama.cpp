# E065 - dev-box phase inventory: what the pp and tg walls are made of (1x gfx1201)

- date: 2026-09-28
- machine: dev-rx9070-16g (1x gfx1201), ROCm 10.0.0, `-ngl 99`
- tier: T1
- status: running
- parent: none (baseline). Reads against E063's bench anchor, E059/E061's kernel work, the
  `GGML_PROF_WINDOW` addition `fc61603f0`, and the removal of llama-bench's implicit window `cc6be7d35`.
- raw: [results/E065-dev-box-phase-inventory/](../results/E065-dev-box-phase-inventory/)

## question and what decides it

With the current defaults on one card: what is a prefill's and a decode step's *wall* made of - host
phases or GPU math - and which kernels own the GPU part? This is an inventory, not an A/B, so no effect
size is claimed and the deciding metric is a *composition*:

- the `[prof]` region table's `pp` / `tg` columns, as a share of the phase wall that the same table
  reports (`phase:prefill` / `phase:decode` totals)
- per-kernel device time from `rocprofv3 --selected-regions`, grouped into the families we track
  (mmq/mul_mat_q on the pp side; mmvq/mmvf and the hc/qsa rows on the tg side)

Prediction, written before running: pp is GPU-bound with a single-digit host share, `graph:compute` at
about the whole wall; tg shows a visible non-GPU share (`phase:sync`, the lazy io regions) with its GPU
part led by the mmvf/mmvq matvecs and the `hc_*` and QSA/indexer rows behind. Falsified if pp's host
share tops ~10% or tg's non-GPU share comes in under ~5%.

Out of scope: no A/B, so no interleaving or effect-size discipline applies; the traced arms are used for
kernel composition only and never for a wall number (E061's rule). The qsa pool mode line is not checked
(needs `-v`), so this round says nothing about pooling.

## arms

Harness is `llama-cli`, so batching is the server's: `-b 2048 -ub 512` are the defaults, i.e. an
8192-token prompt arrives as 16 ubatches of 512.

| arm | env | reps |
| --- | --- | --- |
| pp-untraced | `GGML_PROF_REGIONS=1` | 3 |
| pp-traced | `+ GGML_PROF_WINDOW=pp`, under `rocprofv3 --selected-regions --marker-trace --kernel-trace --stats` | 2 |
| tg-untraced | `GGML_PROF_REGIONS=1` | 3 |
| tg-traced | `+ GGML_PROF_WINDOW=tg`, under the same tracer | 2 |

All arms: `LD_LIBRARY_PATH=$PWD/build/bin:/opt/rocm/lib` (ROCm 10 has no loader entry), one GPU so no
comm path is exercised. Correctness gate: no code is changed and no perf delta is claimed by this round,
so none is required; the golden PPL on this same toolchain is `263100.7437`.

## commands

[`results/E065-dev-box-phase-inventory/commands.sh`](../results/E065-dev-box-phase-inventory/commands.sh)

## harness correction, written before the runs

The pre-registered harness passed the corpus with `-f <corpus> -st`, which does not deliver a prompt.
`-f` sets `common_params::prompt_file` (`common/arg.cpp:1803`) and **nothing in `tools`, `common`, `src`
or `examples` reads that field** - only `imatrix` and `cvector-generator` do. `--stdin` is not registered
for the cli either (`error: invalid argument: --stdin`). `-st`'s own help says the predefined first turn
comes from `--prompt`, and that route works: `-p "$(cat <corpus>)"` measures `tok:prefill 21 calls 32.68k
total`, where the `-f` form measured 2 calls / 4 tokens.

Consequence for earlier data: the user's `results/user/llama-cli-traces/*` runs use the same `-f`
pattern, so any cli trace taken that way profiled decode at a near-empty prompt depth - those logs carry
no `tok:prefill` line that would say otherwise.

Corrected flags, used for everything below: `-c {8192|32768} -b 2048 -ub 1024 -fa 1 -ngl 99 -st --temp 0`
plus `-p "$(cat <corpus>)"`; `-lm/-sm/-ot` are left off because the dummy's memory layout differs from the
real model's. Prompt is the full 32.68k-token corpus at `-c 32768`, the same corpus truncated to the
context at `-c 8192`. Untraced arms: 3 reps per depth per phase. Traced arms: 2 reps per phase at 32768.

Trace CSVs are written to a scratch dir (default `/tmp/pi-coder-scratchpad-obx3lc/e065/traces`) and are
**not committed**: they run to gigabytes on the bench box and `commands.sh` regenerates them. What is
committed is the `[prof]` logs, plus the aggregates in this record. `results/.../E065-*/.gitignore` keeps
`*.csv` out.

## results

(pending)

## reads against

(pending)
