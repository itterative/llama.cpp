# NOTE: this was designed in a different session, using different models

# 16. One-shot (push-based) AllReduce design sketch

Status: DESIGN ONLY, untested -- written without multi-GPU access. To be
validated on the 4-GPU box before any of this lands in
`ggml/src/ggml-cuda/allreduce-p2p.cu`. Companion pieces: the latency model
in `ar_oneshot_sim.py` (calibrated to E73.x measurements) and the E9.x/E73.x
direct-P2P work in `003a-multigpu-allreduce-log.md`.

## Motivation

The sim's calibrated decode model (E73.6: butterfly 48.6 vs ring 41.3 t/s)
decomposes the production per-AR cost at N=4 into ~200 us fixed + ~12.6 us
per event-barrier step. The fixed part is independent of API-call count
(ring issues ~2x butterfly's calls, same fixed cost), pointing at the
event/handshake machinery itself: entry `data_ready` records, cross-device
`cudaStreamWaitEvent` propagation, per-round re-records. With 72 ARs/token
(36 layers x attn_output+ffn_down), that is ~79% of 4-GPU decode token
time. One-shot removes the event machinery entirely: 8 kernel launches per
AR at N=4, zero event API calls.

Projected (sim, not measured): 87-115 t/s 4-GPU decode vs 48.6 baseline;
72-91 t/s at N=2 tensor split (currently layer-split's 74.1 wins there
because tensor-split AR eats the gain).

## Algorithm

Push-based one-shot (TRT-LLM/vLLM custom-AR shape, adapted to this
pipeline's constraints):

1. Each device, on its own compute stream, right after its subgraph
   (program order replaces the entry `data_ready` barrier): a push kernel
   writes its tensor into a dedicated inbox slot on EVERY peer, then a
   doorbell flag in the same peer's flag array.
2. A reduce kernel on the same stream spins on the LOCAL flag array until
   all peers' flags reach the current generation, then sums own tensor +
   inbox slots in fixed device order into the output tensor.

No rounds, no events, no host sync. N=2 degenerates to a single push, i.e.
a 1-round butterfly without the event overhead.

### PCIe-specific constraint: NO remote atomics

System-scope atomics to peer memory over PCIe are unreliable on consumer
root complexes (this box: Ryzen 5800X / Matisse). The design avoids them
entirely:

- Doorbell = plain 32-bit store to peer memory (posted write, always
  supported). PCIe posted writes to the same destination cannot pass each
  other, so flag-after-data ordering holds as long as the issuing SM has
  fenced: every pushing thread executes `__threadfence_system()` after its
  data writes and before the block-done counter bump that eventually
  triggers the flag store.
- Polling = volatile 32-bit loads of LOCAL flags + a local acquire fence
  (`__threadfence()`) after the last flag lands. Local atomics/fences are
  always fine.
- The only atomic anywhere is the LOCAL monotonic block counter
  (`atomicAdd`, device scope) inside the push kernel.

This is THE risk item to validate first on real hardware (see Test plan):
if flag-early arrival ever manifests on this root complex, fall back to an
event-ordered flag kernel; everything else stays.

## Memory layout (per device i, all plain cudaMalloc -- peer-accessible,
same reasoning as dev_tmp in `ggml_cuda_ar_ensure_tmp`)

```
dev_inbox[i]:   (n-1) slots, slot stride = tmp_bytes aligned up to 256 B
                slot(j) = inbox copy of device j's tensor
                (j indexes devices in devices[] order, skipping self)
                +--slot 0--+--slot 1--+--slot 2--+
                | 256B-al. |          |          |
dev_flags[i]:   256 B: n x uint32 generation flags (flag[j] written by
                device j after pushing into dev_inbox[i] slot j), rest pad
dev_counter[i]: 256 B: 1 x uint32 monotonic push-block counter
```

Flags and counter are allocated once at init (cudaMemset 0). Generation is
a host-side `uint32 p->generation++` per AR call, passed as a kernel arg --
flags are never reset, so back-to-back calls cannot alias. Wrap at 2^32:
compare as `(int)(flag - generation) >= 0` if we ever care (4G ARs).

Inbox grows with `ggml_cuda_ar_ensure_tmp`: `inbox_bytes = (n-1) *
tmp_bytes_padded`. Decode payload is 16 KiB, so 48 KiB at N=4 -- nothing.
Prefill-sized tensors make the inbox the largest scratch consumer (3 x
41.9 MB for 27B ub=2048); acceptable, and the auto threshold below keeps
one-shot off that path anyway.

## Kernels (sketch; HIP+CUDA portable, no _system atomics, no libcu++)

Argument marshalling: small POD structs passed by value (kernel param
limit 4 KB; 16 devices x 2 pointer arrays = 256 B, fine). Host builds one
struct per device per launch.

```cpp
#define AR_OS_MAX_DEV 16

struct ar_oneshot_push_args {
    const char * src;                          // my tensor (work_data[i])
    char *       peer_slot[AR_OS_MAX_DEV];     // peer p's inbox slot for me
    unsigned *   peer_flag[AR_OS_MAX_DEV];     // peer p's flag slot for me
    unsigned *   block_done;                   // local monotonic counter
    int          nbytes;                       // padded to 16 B
    int          n;
    unsigned     generation;
};

// grid: (x = copy blocks, y = n-1 target peers). Each block copies its
// slice of src into peer y's slot as uint4 (16 B), fences, bumps the
// counter; the block that completes the generation writes all doorbells.
__global__ void ar_oneshot_push_kernel(ar_oneshot_push_args a) {
    const int p = blockIdx.y < (unsigned)(a.n - 1) ? blockIdx.y : 0; // peer ordinal
    // map peer ordinal p -> device index, skipping self (host pre-bakes
    // peer_slot/peer_flag so that peer_slot[p] is never null here)
    char * dst = a.peer_slot[p];
    const int nvec = a.nbytes >> 4;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec;
         v += gridDim.x * blockDim.x) {
        ((uint4 *) dst)[v] = ((const uint4 *) a.src)[v];
    }
    __threadfence_system();   // data writes visible before the doorbell
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned total = gridDim.x * (a.n - 1);
        const unsigned old = atomicAdd(a.block_done, 1u);
        if ((int) (old - (a.generation - 1) * total) == (int) total - 1) {
            // last block of this call: ring every peer's doorbell
            for (int q = 0; q < a.n - 1; ++q) {
                *((volatile unsigned *) a.peer_flag[q]) = a.generation;
            }
        }
    }
}

struct ar_oneshot_reduce_args {
    char *       dst;                        // my tensor (in-place result)
    const char * self;                       // == dst
    const char * slot[AR_OS_MAX_DEV];        // my inbox, slot[j] from dev j
    const unsigned * flags;                  // my flag array
    int          count;                      // elements
    int          n;
    int          self_idx;
    unsigned     generation;
};

// Single kernel: spin on local flags, then sum in FIXED device order
// (j = 0..n-1, own contribution at j == self_idx) in float, so every
// device executes the identical fp sequence -> N-way bit-identical, same
// property the current add kernel guarantees (allreduce-p2p.cu:118).
template <typename T>
__global__ void ar_oneshot_reduce_kernel(ar_oneshot_reduce_args a) {
    if (threadIdx.x == 0) {
        for (int j = 0; j < a.n; ++j) {
            if (j == a.self_idx) continue;
            while ((int) (((const volatile unsigned *) a.flags)[j] -
                          a.generation) < 0) { /* spin */ }
        }
    }
    __syncthreads();
    __threadfence();          // acquire: inbox reads below see pushed data
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.count;
         i += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < a.n; ++j) {
            const T * src = j == a.self_idx ? (const T *) a.self
                                            : (const T *) a.slot[j];
            acc += (float) src[i];
        }
        ((T *) a.dst)[i] = (T) acc;
    }
}
```

Notes on the sketch:

- Every block spins independently (local flag reads are cheap); a
  single-block-spins-then-others-wait variant is a tuning detail.
- Push copy as uint4 requires nbytes % 16 == 0; f32/f16/bf16 AR tensors
  are n_embd multiples, but keep a scalar tail loop for safety.
- The doorbell block writes ALL peers' flags, so a slow peer link delays
  every flag of this source equally; per-peer doorbells from their
  respective blockIdx.y slices would tighten this -- tune after measuring.
- Spin has no timeout; a dead peer hangs the kernel (same failure
  semantics as a missed event in the current code, but louder). The
  E9.2.5 reboot caveat applies to stress runs.

## Host flow per AR call (replaces the round loops; dispatch site is
`ggml_cuda_ar_allreduce_direct`, allreduce-p2p.cu:1259)

1. `ensure_tmp` as today (+ inbox growth).
2. Memset inactive shards (GGML_TENSOR_FLAG_COMPUTE == 0) as today --
   they push zeros, preserving NCCL semantics.
3. Optional GGML_HIP_AR_BF16 compress as today; one-shot then runs on the
   bf16 working buffer unchanged.
4. `p->generation++`.
5. For each device i (single host loop, same stream model as today):
   build push args (peer pointers via `p->devices`, NO butterfly_order --
   one-shot has no XOR pairing to optimize), launch push kernel on
   `stream_of(i)`, then launch reduce kernel on `stream_of(i)` writing
   `work_data[i]` in place.
6. Optional bf16 decompress as today. Done -- no entry events, no
   per-round records, no exit barrier; the meta-backend's surrounding
   stream sync covers the caller side as it does today.

API calls per AR at N=4: 8 launches (+ rare memsets), vs ~32 for
butterfly. Matches the sim's one-shot host model (~10 cheap calls).

## Integration points in allreduce-p2p.cu

- `ggml_cuda_ar_pipeline_direct` (:75): add `void * dev_inbox[N]`,
  `unsigned * dev_flags[N]`, `unsigned * dev_counter[N]`,
  `size_t inbox_bytes`, `unsigned generation`.
- init (:411): accept `GGML_CUDA_AR_DIRECT_ALGO=oneshot`; allocate +
  zero flags/counter per device (256 B each). Keep peer-access probing
  unchanged -- one-shot needs the same all-pairs access.
- `ggml_cuda_ar_ensure_tmp` (:686): grow `dev_inbox` alongside `dev_tmp`
  (same drain-before-free discipline, :698).
- dispatch (:1248): `algo == ONESHOT` case. `auto`: one-shot below a new
  `GGML_CUDA_AR_DIRECT_ONESHOT_BYTES` (default 256 KiB, conservative),
  butterfly/bde above as today. Gate the whole thing behind opt-in
  (`GGML_CUDA_AR_ONESHOT=1`) until the test plan below has run.
- free (:626): release the three new allocations.

## Numerics

- N-way bit-identical: guaranteed by fixed-order float accumulation
  (reduce kernel loops j ascending on every device), same invariant as
  the current path. No wire-type casts.
- NOT bit-identical to butterfly/bde output: one-shot's summation tree is
  flat, theirs is pairwise. Validation target is the CPU/meta-backend sum
  within fp-reorder tolerance, plus exact equality across the N devices.
- bf16 wire compression composes unchanged (compress -> one-shot on bf16
  -> decompress), inheriting E73.5's max-rel-error numbers.

## Test plan (when the box is available)

1. Doorbell torture FIRST: standalone harness, two GPUs, 1M push/flag
   cycles of varying sizes, verify data checksum matches flag ordering on
   every iteration (validates the PCIe posted-write ordering assumption
   on Matisse). If it ever fails, event-ordered flag fallback.
2. Correctness: N=2 and N=4, sizes 4 KiB-8 MiB, vs meta-backend sum
   (reorder tolerance) and exact cross-device equality; irregular seeds
   per the E73.7 methodology note.
3. Latency sweep -> replace the sim's one-shot column with measurements;
   re-derive OS_PROD_LOW/HIGH.
4. tg sweep (multiple prompts, not one): butterfly vs one-shot at N=4,
   plus N=2 tensor split (does it beat layer-split's 74.1?); compare
   against sim's 87-115 / 72-91 projections.
5. Crossover: one-shot vs bde 256 KiB-4 MiB -> set
   GGML_CUDA_AR_DIRECT_ONESHOT_BYTES from data.
6. Stress: 10k back-to-back ARs, mixed sizes, graphs on
   (GGML_HIP_GRAPHS=ON; AR stays outside capture as today); E9.2.5
   reboot caveat.

## Open risks

- PCIe ordering assumption (item 1 above) -- the whole design rests on
  it; it is the standard assumption NCCL-LL/TRT-LLM make, but those run
  on NVLink more often than bare PCIe.
- The ~200 us fixed cost is a two-point calibration; if the real
  production gap is smaller, one-shot's win shrinks proportionally (the
  sim prints this sensitivity).
- Spin kernels burn SM cycles while peers catch up; fine for decode
  (GPUs idle-wait anyway), wrong if AR ever overlaps compute on-device.
- `auto` policy: if one-shot wins up to ~1 MiB as the sim suggests, the
  existing butterfly/bde crossover logic (:485) becomes the second tier
  of a three-way dispatch.
