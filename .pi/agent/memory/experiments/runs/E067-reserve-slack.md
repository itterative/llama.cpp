# E067 - a reserve slack for the n_kv-proportional tensors: does it kill the ratchet?

- date: 2026-09-28
- machine: dev-rx9070-16g (1x gfx1201), ROCm 10.0.0
- tier: T1
- status: running
- parent: E066 (the trips are size-only, one per 256-token padding step), E065 (what the trips cost), H19
- raw: [results/E067-reserve-slack/](../results/E067-reserve-slack/)

## question and what decides it

`ggml_gallocr_needs_realloc` trips when any non-view tensor's allocation size exceeds what was reserved -
`ggml-alloc.c:1002` is `return talloc->size_max >= node_size`, so a *smaller* graph is always accepted and
only growth trips it. The n_kv-proportional tensors (`blk_cells`, `attn_inp_kq_mask`, the QSA block bias)
grow once per 256-token padding step, and the scheduler answers each growth with a full device synchronize
plus a re-reserve - 1645 of the region's 1689 ms in E065's prefill arm.

Hypothesis: putting a **slack** on the reserved size (a percentage over the measured size) cuts the trips to
about `log_slack(n_kv_max/initial)` instead of one per padding step, and the price is VRAM.

Deciding metric: `sched:realloc_size` call count and the prefill wall from the `[prof]` table, peak VRAM
sampled during the run, and the golden PPL.

Predictions, written before running: at slack 100 (2x) the trips fall from 34 to at most 8; the prefill wall
falls by the overlap the syncs forbid (~0.4 s, so 5481 -> ~5100-5250 ms); peak VRAM rises by at most one
graph-buffer worth (~230 MiB); and **the golden PPL stays bit-identical at 263100.7437**, because only
buffer *sizes* change.

Falsified by: a PPL that moves (the slack touched something it should not), trips that do not fall (so they
are not size-driven after all), or a VRAM rise beyond ~2x of the graph buffers.

## arms

| arm | env | reps |
| --- | --- | --- |
| control | - | 3 |
| slack-50 | `GGML_GALLOCR_RESERVE_SLACK=50` | 3 |
| slack-100 | `GGML_GALLOCR_RESERVE_SLACK=100` | 3 |

Harness as E065/E066: cli, sparse corpus, `-c 32768 -b 2048 -ub 1024 -n 1`, `-v` (the probe is a
`LLAMA_LOG_INFO` line), plus one `llama-perplexity` run on the golden corpus per arm. Gate: the golden PPL
above must not move at all.

## commands

[`results/E067-reserve-slack/commands.sh`](../results/E067-reserve-slack/commands.sh)

## results

**The attempt failed: padding the reserved size alone faults the GPU.** All six slack runs aborted with
`HSA_STATUS_ERROR_MEMORY_FAULT` in `k_set_rows<float, int, __half>` - the get_rows family that consumes
`blk_cells`, the very tensor that grows. So `size_max` is not only a validity threshold: it also feeds the
placement/free-list decisions, while the backing allocation is sized on another path, and padding one
without the other yields tensors that look reserved and are not. The change is reverted and was never
enabled by default.

**The control arm re-measures E065 on the current build and agrees:** 34 `sched:realloc_size` calls,
1689-1693 ms, prefill wall 5500-5773 ms over three reps, `tok:prefill` 32.68k, and
**PPL 263100.7437, bit-identical** - so the harness and the gate are sound and the fault is the change, not
the measurement.

Two harness notes, both mine: the first sweep was invalid because the slack variable went to the VRAM
sampler instead of the measured process (three identical arms is what exposed it), and the PPL runs did not
carry the arm's env, so the recorded PPLs are control values. Fixed in `commands.sh`.

## reads against

- **Negative result, kept.** The slack route needs the same treatment wherever the backing allocation is
  sized - `ggml_gallocr_allocate_node` and the vbuffer - not just in the validity check. H19's literal route
  is the alternative and is now known to be supported: the gallocr accepts any tensor *smaller* than the
  reservation (`ggml-alloc.c:1021` is `size_max >= node_size`), so one reserve that carries the worst-case
  n_kv kills the ratchet, provided the sizes are real.
- A wrong-way allocator change does not fail quietly: it faults the GPU and dumps core.
- E065's numbers reproduce on the newer build, so the ratchet still costs ~8% of prefill and is worth the fix.