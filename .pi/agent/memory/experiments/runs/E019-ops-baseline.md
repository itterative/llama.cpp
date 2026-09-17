# E019 - op-level correctness baseline re-established on ROCm 7.1.1

- date: 2026-09-17 | machine: dev-rx9070-16g (hw v2, ROCm 7.1.1) | tier: T1 | status: done
- command: `test-backend-ops test -b ROCm0` (default `-j 1`), tree at code state 11040, 4m32s
- closes the gap left by the upgrade: B1's "1500/1500 pass" was measured on ROCm 6.4.4 (E001)

## baseline

**5641 cases: 5633 OK, 7 FAIL, 0 SKIP.**

All 7 failures are `FLASH_ATTN_EXT(hsk=192,hsv=128,...)` with `ERR` between 0.0084 and 0.0250
against a 0.0005 tolerance - the pre-existing family the backlog hygiene note already marks as
"not our shape, do not chase". Our shape is clean: **142/142 `hsk=256,hsv=256` cases pass**.

One discrepancy worth keeping visible rather than silent: the backlog records **6** known FA
failures at this shape and we see **7**, in the same magnitude band. Either the suite gained a
case since that note or the old count was off by one. Not chased; if this baseline is ever used
to claim "no new failures", the number to compare is 7.

So the gate that rule 3 requires exists on the current stack, and E001's conclusion survived the
upgrade.

## side result: graph capture works locally now

9430 `ggml_backend_cuda_graph_compute: CUDA graph warmup complete` messages during the run, i.e.
capture is active per test case on gfx1201 / ROCm 7.1.1. Under 6.4.4 it never succeeded here, and
the hw profile v2 note predicted this as the one upside of the upgrade. Consequence: capture and
replay behaviour (E008/E010-class probes) is now a **T1** question, not bench-only.

## trap: `-j > 1` aborts the run

`test-backend-ops test -b ROCm0 -j 8` died after ~22 s with

```
ROCm error: operation would make the legacy stream depend on a capturing blocking stream
ROCm error: operation failed due to a previous error during capture
```

Worker threads issue work on the device while another thread is mid-capture, which invalidates the
capture, and the run aborts rather than degrading. It had printed 17k lines first, so a partial
log looks like a healthy run. **Use `-j 1`.** Related caution: this is a reminder that capture on
this backend is fragile in ways we have not looked for, and the whole fixed-cost story (E010/E011)
assumes capture is working.
