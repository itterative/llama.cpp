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
