# E033 - the bench box GPU hangs, and why I stopped blaming the sparse kernel

Three incidents on the bench box while chasing the sparse FA flag: two process aborts inside ROCR's
`HwExceptionHandler` and one box that needed a power cycle. My running theory was candidate R1 from
E032 - an unclamped gather index becoming an address. `dmesg` kills it.

```
amdgpu 0000:19:00.0: MES(0) failed to respond to msg=REMOVE_QUEUE
amdgpu 0000:19:00.0: MES(0) failed to respond to msg=SUSPEND
amdgpu 0000:19:00.0: MES might be in unrecoverable state, issue a GPU reset
amdgpu 0000:19:00.0: MODE1 reset ... GPU reset succeeded, trying to resume
amdgpu 0000:19:00.0: [drm] device wedged, but no recovery needed
```

Three resets at 5448, 5614 and 5623 seconds since boot, always the same device, always during **queue
teardown**, and **not one GPUVM fault line anywhere**. A bad address would have been reported by the
driver as a VM fault with the address; what happened instead is that the scheduler stopped answering
and the kernel reset the card. That is a timeout, not an illegal access, and userspace saw it as
`HW Exception by GPU node-4 ... reason: GPU Hang`.

The sequence that fits: I recommended `HIP_LAUNCH_BLOCKING=1` to get synchronous attribution. On a
four-card box with a spin-style p2p allreduce that serialises launches, that is a way to *create* a
deadlock - the run pinned one card at 100% utilisation and ~80 W for minutes with no progress. Killing
it left a wedged gang; the driver reset the card and declared `no recovery needed`; the next process
then died on that card before doing anything, at `n_ctx_seq = 4608`, where the sparse gate cannot be
open because it needs 4102 used KV. So the incident I was about to attribute to my kernel happened
before my kernel could have run.

After a power cycle, the full matrix completed twice on build 11071, dense and sparse, including 131k
with the flag set (E027 correction 3). No fault, no hang.

Status: not attributed, and deliberately not closed. What is *not* supported any more is the
"sparse FA faults on the bench" framing that this branch's memory carried for a day. If it returns,
the useful first question is whether sparse could even have been engaged at the depth where it died,
and the second is whether the box had already been reset once in that boot.

Lessons worth keeping:
- `HIP_LAUNCH_BLOCKING=1` is unsafe as a diagnostic on multi-GPU spin-sync paths. It does not just
  slow things down, it can deadlock the device and then poison the boot.
- Utilisation and power disagree; power wins. 100% util at 80 W is a queue that is resident and
  going nowhere, which is what a hang looks like.
- `AMD_LOG_LEVEL=2` emitted nothing on this ROCm build, so it is not a way to name a faulting kernel here.
- A GPU *hang* produces no `gpucore.*` and the kernel devcoredump is single-shot: the second and
  third resets destroyed the only artifact from the first.

## Update 2026-09-24: power profile changed, validation pending

The hangs surfaced again on the 4-card box (E045's server crash at ctx 245760, and during the E050
suite), still with the same `MES failed to respond to msg=REMOVE_QUEUE / SUSPEND` signature. The user
has switched the box's **power profile from `auto` to `compute`** as a possible fix; untested as of
writing, and recorded in the hw profile as `bench-4x-r9700-32g` v2.

What would count as evidence, since "it hasn't hung today" is not: the failures clustered on long
`draft-mtp` sessions at the deepest context, so the test is the workload that previously died (MTP
decode at ctx ~245760, an hour or more, or the E050 bench suite end to end) running clean more than
once. If it does hold, note that a profile change is a comparability break for power-bound work: any
`t/s` comparison straddling 2026-09-24 needs the v1/v2 boundary stated, and `tg` at these depths is
plausibly power-limited enough to move.

Nothing here closes the two live suspects, both still unproven: VRAM headroom at ctx 245760 (free
memory was 22832 MiB aggregate after load, and `common_fit_params` refuses to fit under tensor split,
so nothing would back the context off automatically), and whatever wedges MES in the first place. A
power-profile change addresses neither directly, so a clean run would be a clue rather than a diagnosis
- most likely pointing at clock/DVFS transitions on the idle-to-burst pattern of speculative decode
rather than at memory.
