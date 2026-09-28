# E074 - bench box: dense prefill does not re-reserve, and the fusion is worth 8.75% of pp

Status: **closed**. T2 data from the user's runs on `bench-4x-r9700-32g`, build `ede6d511c` (the H19 fix is
in), `llama-bench`, `-lm none -sm tensor -fa 1 -lzm on-direct -ot per_layer_token_embd=CPU -b 2048 -ub 1024`.
Raw logs (gitignored by `results/user/.gitignore`, referenced here): `results/user/h19-runs-ede6d511c/`.

## 1. Reference run vs the 2026-09-27 baseline: no change

Same flag set as `bench-finish-2026-09-27/ref-baseline-u4-r10.log`, which predates the env-default flips.
Per-test deltas, new against old: pp512 -0.5..-1.3%, pp4096 -0.2..-0.8%, pp8192 -0.2..-0.8%, tg128 -0.1..+0.2%.
All inside the established pp cross-session agreement (~1.2%) - so the flips plus the fix did not move this
table, and nothing regressed.

## 2. The two prof arms (the decisive part)

Both `GGML_PROF_REGIONS=1`, both pp-only (`pp4096 @ d16384`), identical except `GGML_CUDA_DISABLE_FUSION=1`:

| row | fusion on | fusion off | delta |
| --- | --- | --- | --- |
| pp4096 t/s | 2138.37 ± 3.32 | 1951.32 ± 3.98 | **-8.75%** |
| `phase:prefill` | 28 calls, 21861.69 ms | 28 calls, 23896.35 ms | +9.31% |
| `graph:compute` | 56 calls, 15604.58 ms | 56 calls, 17641.90 ms | +13.06% |
| `phase:sync` | 22 calls, 4296.86 ms | 22 calls, 4812.76 ms | +12.01% |
| `graph:alloc` | 56 calls, 1146.16 ms | 56 calls, 1170.54 ms | +2.13% |
| `graph:build` | 56 calls, 56.42 ms | 56 calls, 56.87 ms | +0.80% |
| `mctx:apply` | 56 calls, 15.41 ms | 56 calls, 15.62 ms | +1.41% |
| `sched:realloc_*` | **absent** | **absent** | - |

### 2a. The dense prefill on that box does not re-reserve

`grep -c '^\[prof\].*sched:realloc'` is **0 in both arms**, i.e. the region recorded no calls in either
process. Calibration: on the dev box the same region is absent with the fix and shows 34 calls / 1699 ms
without it, so an absent row means genuinely none. This is consistent with the user's own E057/H22 note
("0 for dense" against 19-21 per 7.5k cells for pinned or gapped).

**Consequence, and a retraction:** the fix is neutral for the dense prefill on the bench box. The earlier
estimate of "seconds per prompt on 4 cards" was the dev-box 34 trips priced at E056's ~300 ms per re-reserve;
that does not transfer to this workload. The fix's T2 exposure is the paths that do re-reserve - the
non-dense/pinned/vision case (H22) and the request path, where the dev box's `llama-cli` prompt still measured
34 - and none of that is in these logs, since `llama-bench` is not the request path.

### 2b. The fusion is worth 8.75% of prefill

Fusion off costs 187 t/s at `pp4096 @ d16384`, i.e. 291 ms per 4096-token pass (71 us/token), and the region
split says it is compute: `graph:compute` +2037 ms over the run, `phase:sync` +516 ms, while `graph:alloc`,
`graph:build` and `mctx:apply` are unchanged. This is the measurement behind keeping the fusion in the H19 fix
rather than disabling it: had we disabled it, prefill would have lost this 8.75% permanently.

Still unmeasured: the fusion's **decode** value. These runs are pp-only (`-n 0`), and the reference table's
`tg128 @ d16384 = 43.11` is a fusion-on number. A `-n 128` pair with and without the env var would close it,
and it is the side where a MoE weighted reduction should matter most.

### 2c. The normal allocation cost is untouched

`graph:alloc` is ~20.5 ms/call over 56 calls = 1.15 s, i.e. ~5% of the prefill total, with **no** re-reserves
involved. That is the same ~21 ms/call figure E062 used for H19's decode line, so it is a separate cost: it is
the per-batch scheduler allocation itself, and the H19 fix does not address it. Worth keeping on the backlog as
such.

## 3. Open observation from the user

Prefill **GPU utilization looks higher than remembered**. What the data can say:

- Not attributable to the H19 fix in these runs: there were no re-reserves to remove in either arm.
- Not visible as throughput against the 2026-09-27 baseline either (section 1 is flat), so a utilization change
  versus *that* session is measurement conditions rather than a code change.
- Versus further back, the plausible cause is the prefill-pooling work (E056: pp8192 1781 t/s with
  prefill-pooling excluded vs 2230 pooled), which is more real work per second and predates the baseline.
- To measure instead of remember: `rocprofv3 --kernel-trace` with the branch's `GGML_PROF_WINDOW=pp` and sum
  kernel time per pass against the wall. Note `graph:compute` is the async entry point, so it is not device
  time - the tracer is the only direct source. The tracer's cross-arm caveat (E061) does not apply to a
  within-run utilisation ratio.

## Caveats

Arms are separate invocations, not interleaved: llama-bench's own bands are ±0.16% and ±0.20% here and the
fusion effect is 8.75%, so it is well resolved, but a paired repeat would make it airtight. One test only
(pp), one depth.