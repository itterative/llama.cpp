#include "allreduce-p2p.cuh"
#include "convert.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// Env-var helper for the N-GPU direct-P2P pipeline (runs on HIP and CUDA;
// allreduce-host.cu keeps its own copy of the same helper for its CUDA-only
// pipeline, since the two files no longer share a translation unit).
static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)

// ---------------------------------------------------------------------------
// Direct-P2P AllReduce (N-GPU). Runs on HIP and CUDA. Each pair of GPUs
// exchanges data with cuda/hipMemcpyPeerAsync (device-to-device, no host
// staging). Requires peer access on every ordered pair; init probes this and
// returns nullptr otherwise, so a failed init never beats the meta-backend
// fallback. Tensor types F32/F16/BF16, same precision in/out.
//
// Three algorithms (GGML_CUDA_AR_DIRECT_ALGO, default "auto"):
//   butterfly - recursive-doubling: round r pairs device i with i^(1<<r);
//     after log2(N) rounds every device holds the full sum. Moves log2(N)*size
//     per device. Power-of-2 N only. Fewest steps -> best when latency-bound.
//   ring      - reduce-scatter + all-gather over N chunks. Any N >= 2. Moves
//     2*(N-1)/N*size per device (least bandwidth) but 2*(N-1) sequential steps.
//   bde       - Rabenseifner: recursive-halving reduce-scatter +
//     recursive-doubling all-gather over the same XOR network as butterfly.
//     Power-of-2 N only. Same bytes moved as ring in 2*log2(N) steps, so it
//     supersedes ring for power-of-2 N.
//
// auto picks per call: butterfly below GGML_CUDA_AR_DIRECT_AUTO_RING_BYTES
// (default 1 MiB, latency-bound), bde at/above it (bandwidth-bound); ring only
// when N is not a power of 2. N=2 always uses butterfly.
//
// dev_tmp scratch holds the whole tensor and grows on demand
// (GGML_CUDA_AR_DIRECT_TMP_BYTES sets the initial size, not a cap); only an
// allocation failure falls back to the meta-backend.
//
// Device ordering is chosen at init from a per-directed-pair bandwidth/latency
// probe (GGML_CUDA_AR_DIRECT_RING_ORDER=auto|identity|<permutation>), which
// minimizes the ring's bottleneck edge on hardware with asymmetric per-pair
// P2P bandwidth; at N=4 the same probe seeds butterfly/bde's mapping (their
// 2-round XOR graph is isomorphic to a 4-cycle). See ggml_cuda_ar_ring_pick_order.
//
// The meta-backend calls this between per-device graph_compute calls, never
// inside a graph capture, so the event handshakes are not captured.
//
// GGML_HIP_AR_BF16 (build flag, off by default): compress F32 tensors to bf16
// before the round loop, run every round on the bf16 buffer (half the bytes),
// decompress once after the last round. F16/BF16 tensors skip it (already 2
// bytes/elem). GGML_CUDA_AR_DIRECT_BF16_THRESHOLD (default 1 MiB) skips
// compression for F32 tensors below that size, where the extra compress/
// decompress launches outweigh the halved transfer; 0 = always compress.
// ---------------------------------------------------------------------------

enum ggml_cuda_ar_algo {
    GGML_CUDA_AR_ALGO_BUTTERFLY = 0,
    GGML_CUDA_AR_ALGO_RING      = 1,
    GGML_CUDA_AR_ALGO_AUTO      = 2,
    GGML_CUDA_AR_ALGO_BDE       = 3,
    GGML_CUDA_AR_ALGO_ONESHOT   = 4,
};

struct ggml_cuda_ar_pipeline_direct {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   tmp_bytes;          // per-device scratch size (grows to max AR size)
    bool     tmp_alloc_failed;   // set once a growth fails; pipeline dead after
    ggml_cuda_ar_algo algo;      // from GGML_CUDA_AR_DIRECT_ALGO
    size_t   auto_ring_bytes;    // algo=auto: bde at/above this, butterfly below

    // ring_order[pos] / butterfly_order[L] = index into devices[] sitting at
    // ring position pos / butterfly logical position L. Identity unless set by
    // the init-time bandwidth probe (butterfly_order: N=4 only). pairs_in_round
    // stays in the untranslated XOR index space (needed for bde's bit math);
    // only actual resource accesses go through these maps. See
    // ggml_cuda_ar_ring_pick_order and ggml_cuda_ar_pipeline_direct_init.
    int      ring_order[GGML_CUDA_MAX_DEVICES];
    int      butterfly_order[GGML_CUDA_MAX_DEVICES];

    // Per-device scratch for the per-pair copy. All AR work runs on each
    // device's compute stream (ctx->stream()); there is no dedicated AR stream.
    void *   dev_tmp[GGML_CUDA_MAX_DEVICES];

#if defined(GGML_HIP_AR_BF16)
    // Half-sized bf16 working buffers, used for F32 tensors when compression is
    // active: dev_bf16 holds the compressed running sum, dev_tmp_bf16 is the
    // bf16 counterpart of dev_tmp.
    void *   dev_bf16[GGML_CUDA_MAX_DEVICES];
    void *   dev_tmp_bf16[GGML_CUDA_MAX_DEVICES];
    size_t   bf16_threshold;     // F32 tensors below this skip compression; 0 = always
#endif

    // One event per device (cudaEventDisableTiming). data_ready[i] = "device
    // i's work buffer is safe for peers to read"; recorded at entry and after
    // each round's phase-2 add, waited on before a peer's next-round copy. The
    // ping-pong (phase-1 reads one buffer, phase-2 writes the other) needs no
    // within-round barrier; this carries the cross-round dependency.
    // Unused by the one-shot path, which has no cross-round dependency.
    cudaEvent_t data_ready[GGML_CUDA_MAX_DEVICES];

    // One-shot path: per-device inbox, n slots x 2 generation parities x 2 bytes
    // of wire per byte of payload (LL tag). Allocated with the same growth path
    // as dev_tmp so the slot layout is a pure function of tmp_bytes.
    void *   os_inbox[GGML_CUDA_MAX_DEVICES];
    unsigned os_generation;      // host-side tag, +1 per collective
    size_t   os_bytes;           // GGML_CUDA_AR_DIRECT_ONESHOT_BYTES: above it, fall back to auto's pick

    // Butterfly/bde pairing: pairs_in_round[r][2k+0..1] = (a, b), log2(N) rounds.
    int      n_rounds;
    int      n_pairs_in_round[16];
    int      pairs_in_round[16][2 * GGML_CUDA_MAX_DEVICES];
};

// One-shot inbox size for a given tmp high-water mark: n source slots x 2
// generation parities x 2 wire bytes per payload byte (the LL tag).
static size_t ggml_cuda_ar_os_bytes(const ggml_cuda_ar_pipeline_direct * p, size_t tmp_bytes) {
    return (size_t) p->n_devices * 2 * 2 * tmp_bytes;
}

// In-place add dst[i] += src[i], summed in float, written back as T. Peer data
// is copied losslessly as raw bytes, so every device does the same float add on
// the same inputs and all N land on identical bits -- no cast-through-wire-type
// round-trip (that would break N-way consistency, not fix it).
template <typename T>
static __global__ void ggml_cuda_ar_direct_add_kernel(
        T       * __restrict__ dst,
        const T * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        dst[i] = (T) (((float) dst[i]) + ((float) src[i]));
    }
}

// ---------------------------------------------------------------------------
// One-shot AllReduce: ONE kernel launch per device per collective.
//
// The copy-based algorithms above issue ~4 API calls per device per round, so
// at N=4 a collective costs ~30 host calls against NCCL's 6 -- measured 127 us
// vs 37 us of host time per call on 4x R9700. Host cost is linear in API calls
// at ~4-5 us each, so anything that wants to beat NCCL has to get under 6.
//
// Wire unit is 8 bytes: 4 bytes of payload plus a 4 byte generation tag in the
// same store. An aligned 64-bit store is atomic, so a reader sees either the
// old unit or the new one -- no flag array, no arrival counter, no system
// fence, and the posted-write-ordering assumption the doorbell design rested on
// never comes up. Bytes on the wire double, which is irrelevant at 10 KB.
//
// Each thread pushes exactly the units it later reduces, so the in-place write
// back into my own tensor cannot race with another thread pushing that unit.
// Every device launches the same grid, so thread t covers the same units on
// every device, and a thread has pushed its units before it waits on any peer.
// That makes the grid a correctness constraint: a block that never launches
// never pushes, and peers would then spin on units that do not exist. Hence the
// fixed co-resident grid below (16 blocks on a 54-CU GPU).
//
// Inboxes are double-buffered by generation parity. A peer cannot be two
// generations ahead while I am still reading generation g: its g+1 kernel is
// enqueued on its stream after its g kernel, and that one needed my g push.
// ---------------------------------------------------------------------------

#define GGML_CUDA_AR_OS_BLOCKS  16
#define GGML_CUDA_AR_OS_THREADS 256
#define GGML_CUDA_AR_OS_SPIN    200000000L

struct ggml_cuda_ar_os_args {
    const unsigned *         self;                          // my tensor, as 4 B units
    unsigned *               dst;                           // == self, in place
    unsigned long long *     to  [GGML_CUDA_MAX_DEVICES];   // peer j's inbox slot for my data
    const unsigned long long * from[GGML_CUDA_MAX_DEVICES]; // my inbox slot holding peer j's data
    unsigned gen;
    int      n;
    int      self_i;
    int      nunits;
    bool     compute;                                       // false: I contribute zeros
};

// K * sizeof(T) == 4: one f32 per unit, or two f16/bf16.
template <typename T, int K>
static __global__ void ggml_cuda_ar_oneshot_kernel(ggml_cuda_ar_os_args a) {
    union os_unit {
        unsigned raw;
        T        v[K];
    };

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;

    // Push my slice into every peer's inbox. Posted writes, no waiting.
    for (int u = tid; u < a.nunits; u += nth) {
        const unsigned raw = a.compute ? a.self[u] : 0u;
        const unsigned long long w = ((unsigned long long) raw << 32) | (unsigned long long) a.gen;
        for (int j = 0; j < a.n; ++j) {
            if (j != a.self_i) {
                a.to[j][u] = w;
            }
        }
    }

    // The 8 B store makes a peer's read atomic -- it sees the old unit or the
    // new one -- but atomicity is not visibility: without this fence the writes
    // can sit in this device's cache hierarchy while the thread waits on its
    // peers, which are waiting for exactly those writes.
    __threadfence_system();

    // Reduce in ascending device order on every device, in float, so all N
    // devices land on identical bits -- same invariant the add kernel keeps.
    for (int u = tid; u < a.nunits; u += nth) {
        float acc[K] = {};

        for (int j = 0; j < a.n; ++j) {
            unsigned raw;

            if (j == a.self_i) {
                raw = a.compute ? a.self[u] : 0u;
            } else {
                unsigned long long w    = 0;
                long               spin = 0;

                // volatile: the value is written by another device, so the load
                // must not be hoisted out of the wait loop.
                const volatile unsigned long long * slot = a.from[j];

                while (true) {
                    w = slot[u];
                    const unsigned tag = (unsigned) w;
                    if (tag == a.gen) {
                        break;
                    }
                    if ((int) (tag - a.gen) > 0 && spin++ == 0) {
                        printf("ar-oneshot: peer %d slot %d is at tag %u, wanted %u\n", j, u, tag, a.gen);
                    }
                    if (spin > GGML_CUDA_AR_OS_SPIN) {
                        printf("ar-oneshot: timeout waiting for peer %d unit %d (tag %u, gen %u)\n",
                               j, u, (unsigned) w, a.gen);
                        break;
                    }
                }

                raw = (unsigned) (w >> 32);
            }

            os_unit in;
            in.raw = raw;
            for (int k = 0; k < K; ++k) {
                acc[k] += (float) in.v[k];
            }
        }

        os_unit out;
        for (int k = 0; k < K; ++k) {
            out.v[k] = (T) acc[k];
        }
        a.dst[u] = out.raw;
    }
}

// GGML_CUDA_AR_ONESHOT_DEBUG=1: dump every pointer the first collective passes
// to the kernel, with what the runtime thinks each one is. The box reported a
// write fault at a low VA, so this is what pins down which of the four it was.
static void ggml_cuda_ar_os_dump_ptr(const char * what, const void * ptr) {
    hipPointerAttribute_t at = {};
    const hipError_t e = hipPointerGetAttributes(&at, ptr);

    fprintf(stderr, "[ar-os]   %-22s %p  err=%d type=%d dev=%d managed=%d devptr=%p\n", what, ptr,
            (int) e, (int) at.type, at.device, (int) at.isManaged, at.devicePointer);
    fflush(stderr);
}

static void ggml_cuda_ar_launch_oneshot(
        ggml_cuda_ar_pipeline_direct * p,
        void                        ** work_data,
        ggml_type                      work_type,
        int64_t                        ne,
        const bool                   * compute,
        cudaStream_t                 * streams) {
    const int    n          = p->n_devices;
    const size_t type_size  = ggml_type_size(work_type);
    const int    nunits     = (int) ((size_t) ne * type_size / 4);
    const size_t slot_bytes = 2 * p->tmp_bytes;
    const unsigned gen      = ++p->os_generation;
    const int    parity     = (int) (gen & 1);

    ggml_cuda_ar_os_args a = {};
    a.n       = n;
    a.gen     = gen;
    a.nunits  = nunits;

    for (int i = 0; i < n; ++i) {
        a.self_i  = i;
        a.self    = (const unsigned *) work_data[i];
        a.dst     = (unsigned *)       work_data[i];
        a.compute = compute[i];

        for (int j = 0; j < n; ++j) {
            if (j == i) {
                a.to[j]   = nullptr;
                a.from[j] = nullptr;
                continue;
            }
            a.to[j]   = (unsigned long long *) ((char *) p->os_inbox[j] + ((i * 2 + parity) * slot_bytes));
            a.from[j] = (const unsigned long long *) ((const char *) p->os_inbox[i] + ((j * 2 + parity) * slot_bytes));
        }

        ggml_cuda_set_device(p->devices[i]);

        static const bool dbg = getenv("GGML_CUDA_AR_ONESHOT_DEBUG") != nullptr;
        if (dbg && gen <= 2) {
            fprintf(stderr, "[ar-os] dev %d gen %u nunits %d slot %zu tmp %zu\n",
                    p->devices[i], gen, nunits, slot_bytes, p->tmp_bytes);
            ggml_cuda_ar_os_dump_ptr("self",  a.self);
            ggml_cuda_ar_os_dump_ptr("dst",   a.dst);
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const std::string tag = "to[dev" + std::to_string(p->devices[j]) + "]";
                const std::string fag = "from[dev" + std::to_string(p->devices[j]) + "]";
                ggml_cuda_ar_os_dump_ptr(tag.c_str(), a.to[j]);
                ggml_cuda_ar_os_dump_ptr(fag.c_str(), a.from[j]);
            }
        }

        switch (work_type) {
            case GGML_TYPE_F32: ggml_cuda_ar_oneshot_kernel<float, 1><<<GGML_CUDA_AR_OS_BLOCKS, GGML_CUDA_AR_OS_THREADS, 0, streams[i]>>>(a); break;
            case GGML_TYPE_F16: ggml_cuda_ar_oneshot_kernel<half,  2><<<GGML_CUDA_AR_OS_BLOCKS, GGML_CUDA_AR_OS_THREADS, 0, streams[i]>>>(a); break;
            default:            ggml_cuda_ar_oneshot_kernel<nv_bfloat16, 2><<<GGML_CUDA_AR_OS_BLOCKS, GGML_CUDA_AR_OS_THREADS, 0, streams[i]>>>(a); break;
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool ggml_cuda_ar_ensure_tmp(ggml_cuda_ar_pipeline_direct * p, size_t need_bytes);

// GGML_CUDA_AR_ONESHOT_PROBE=<iterations>: run the one-shot over the real links
// before any model work. Each device seeds every element with (rank+1), so the
// result must be n*(n+1)/2 on every device; that checks the 8-byte wire unit
// (one atomic store carrying payload plus generation) on the actual root
// complex instead of trusting the assumption, and it also times one round trip
// with no NCCL in the loop. On failure the pipeline drops back to auto.
static void ggml_cuda_ar_oneshot_probe(ggml_cuda_ar_pipeline_direct * p) {
    const int iters = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_ONESHOT_PROBE", 0);

    if (iters <= 0 || p->algo != GGML_CUDA_AR_ALGO_ONESHOT) {
        return;
    }

    const int     n   = p->n_devices;
    const int64_t ne  = 1024;
    float       * host = new float[ne];

    if (!ggml_cuda_ar_ensure_tmp(p, (size_t) ne * sizeof(float))) {
        delete[] host;
        return;
    }

    void       * work_data[GGML_CUDA_MAX_DEVICES];
    bool         compute[GGML_CUDA_MAX_DEVICES];
    cudaStream_t streams[GGML_CUDA_MAX_DEVICES];

    for (int i = 0; i < n; ++i) {
        work_data[i] = p->dev_tmp[i];
        compute[i]   = true;
        streams[i]   = 0;

        ggml_cuda_set_device(p->devices[i]);
        for (int64_t e = 0; e < ne; ++e) {
            host[e] = (float) (i + 1);
        }
        CUDA_CHECK(cudaMemcpy(p->dev_tmp[i], host, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice));
    }

    if (getenv("GGML_CUDA_AR_ONESHOT_DEBUG") != nullptr) {
        fprintf(stderr, "[ar-os] pre-launch: n=%d ne=%d tmp=%zu slot=%zu inbox=%zu\n",
                n, (int) ne, p->tmp_bytes, 2 * p->tmp_bytes, (size_t) n * 2 * 2 * p->tmp_bytes);
        for (int i = 0; i < n; ++i) {
            const std::string tag = "dev" + std::to_string(p->devices[i]);
            ggml_cuda_ar_os_dump_ptr((tag + " dev_tmp").c_str(),  p->dev_tmp[i]);
            ggml_cuda_ar_os_dump_ptr((tag + " os_inbox").c_str(), p->os_inbox[i]);
        }
        fflush(stderr);
    }

    ggml_cuda_ar_launch_oneshot(p, work_data, GGML_TYPE_F32, ne, compute, streams);
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float expect = 0.0f;
    for (int i = 0; i < n; ++i) {
        expect += (float) (i + 1);
    }

    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaMemcpy(host, p->dev_tmp[i], (size_t) ne * sizeof(float), cudaMemcpyDeviceToHost));
        for (int64_t e = 0; e < ne; ++e) {
            ok = ok && host[e] == expect;
        }
    }

    // Timing loop: re-running without re-seeding keeps the ping-pong advancing
    // through generations, which is the part that would break if the double
    // buffering were wrong.
    const auto t0 = std::chrono::steady_clock::now();

    for (int k = 0; k < iters; ++k) {
        ggml_cuda_ar_launch_oneshot(p, work_data, GGML_TYPE_F32, ne, compute, streams);
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    const double lat_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count() / (double) iters;

    // Back-to-back collectives with a single sync at the end. This is the number
    // that is comparable to meta:allreduce: the loop above pays n device syncs
    // per iteration, which swamps the collective itself.
    const auto t1 = std::chrono::steady_clock::now();

    for (int k = 0; k < iters; ++k) {
        ggml_cuda_ar_launch_oneshot(p, work_data, GGML_TYPE_F32, ne, compute, streams);
    }

    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const double thr_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t1).count() / (double) iters;

    if (ok) {
        // stderr, not GGML_LOG_INFO: llama-bench mutes INFO unless verbose is on, and
        // this is the only measurement of one round trip over the real links.
        fprintf(stderr, "[ar-os] probe ok over %d GPUs: %.1f us pipelined per collective, %.1f us\n"
                        "        with a full device wait each round; %d KB payload, sum %.1f identical\n",
                n, thr_us, lat_us, (int) (ne * 4 / 1024), expect);
        fflush(stderr);
    } else {
        GGML_LOG_ERROR("%s: probe FAILED, expected sum %.1f; dropping to algo=auto\n", __func__, expect);
        p->algo = GGML_CUDA_AR_ALGO_AUTO;
    }

    delete[] host;
}

// Measurement-based ring shape (GGML_CUDA_AR_DIRECT_RING_ORDER, default
// "auto"): time a small and a large P2P copy on every ordered pair, fit
// t(bytes) = latency + bytes/bandwidth per directed edge, and pick the cyclic
// order minimizing the bottleneck edge time at a reference chunk size (a ring
// step is gated by its slowest edge, so route around the worst directed edges
// on hardware with asymmetric per-pair P2P bandwidth). Requires p->dev_tmp and
// peer access already set up; called from init.
static void ggml_cuda_ar_ring_pick_order(ggml_cuda_ar_pipeline_direct * p) {
    const int n = p->n_devices;

    // GGML_CUDA_AR_DIRECT_RING_ORDER=auto (default, measure) | identity |
    // comma-separated permutation of device-list indices (e.g. "0,3,1,2").
    const char * order_env = getenv("GGML_CUDA_AR_DIRECT_RING_ORDER");
    if (order_env && order_env[0] != '\0' && strcmp(order_env, "auto") != 0) {
        if (strcmp(order_env, "identity") == 0) {
            return;  // ring_order is already identity
        }
        int  list[GGML_CUDA_MAX_DEVICES];
        int  count = 0;
        bool ok = true;
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", order_env);
        for (char * tok = strtok(buf, ","); tok != nullptr; tok = strtok(nullptr, ",")) {
            if (count >= n) { ok = false; break; }
            list[count++] = atoi(tok);
        }
        bool seen[GGML_CUDA_MAX_DEVICES] = {};
        for (int k = 0; ok && k < count; ++k) {
            if (list[k] < 0 || list[k] >= n || seen[list[k]]) { ok = false; }
            seen[list[k]] = true;
        }
        if (ok && count == n) {
            for (int k = 0; k < n; ++k) {
                p->ring_order[k] = list[k];
            }
            GGML_LOG_INFO("%s: ring order from GGML_CUDA_AR_DIRECT_RING_ORDER: %s\n",
                          __func__, order_env);
        } else {
            GGML_LOG_WARN("%s: invalid GGML_CUDA_AR_DIRECT_RING_ORDER '%s' "
                          "(need a permutation of 0..%d); using identity\n",
                          __func__, order_env, n - 1);
        }
        return;
    }

    // Probe sizes: one latency-dominated, one bandwidth-dominated. The large
    // probe is capped by tmp_bytes (already clamped to >= 1 MiB at init).
    const size_t probe_large = p->tmp_bytes < (size_t) 8 * 1024 * 1024 ? p->tmp_bytes
                                                                       : (size_t) 8 * 1024 * 1024;
    const size_t probe_small = probe_large / 64;
    const int    n_warm = 2, n_iter = 8;

    double lat_us[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};
    double bw_Bps[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};

    // Time one directed P2P copy dev[dst] <- dev[src] of `bytes`, averaged over
    // n_iter post-warmup runs. Returns microseconds, or a negative value if the
    // copy failed (already logged, sticky error cleared).
    auto probe = [&](int dst, int src, size_t bytes) -> double {
        ggml_cuda_set_device(p->devices[dst]);
        double t_us = 0;
        for (int it = 0; it < n_warm + n_iter; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            cudaError_t e = cudaMemcpyPeerAsync(p->dev_tmp[dst], p->devices[dst],
                                                p->dev_tmp[src], p->devices[src],
                                                bytes, (cudaStream_t) 0);
            if (e == cudaSuccess) {
                e = cudaDeviceSynchronize();
            }
            const auto t1 = std::chrono::steady_clock::now();
            if (e != cudaSuccess) {
                GGML_LOG_WARN("%s: probe copy dev %d <- dev %d failed: %s; "
                              "keeping identity ring order\n",
                              __func__, p->devices[dst], p->devices[src],
                              cudaGetErrorString(e));
                (void) cudaGetLastError();
                return -1.0;
            }
            if (it >= n_warm) {
                t_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
        }
        return t_us / n_iter;
    };

    for (int d = 0; d < n; ++d) {
        for (int s = 0; s < n; ++s) {
            if (s == d) {
                continue;
            }
            const double t_small = probe(d, s, probe_small);
            if (t_small < 0) return;
            const double t_large = probe(d, s, probe_large);
            if (t_large < 0) return;

            // Fit t(bytes) = lat + bytes/bw through the two probe points.
            // Times include host launch/sync overhead, but that is uniform
            // across edges, so the comparison below stays valid.
            if (t_large > t_small) {
                bw_Bps[d][s] = (double) (probe_large - probe_small) / (t_large - t_small) * 1e6;
                lat_us[d][s] = t_small - (double) probe_small / bw_Bps[d][s] * 1e6;
                if (lat_us[d][s] < 0) {
                    lat_us[d][s] = 0;
                }
            } else {  // noise; degrade to a bandwidth-only estimate
                bw_Bps[d][s] = (double) probe_large / t_large * 1e6;
                lat_us[d][s] = 0;
            }
            GGML_LOG_INFO("%s: P2P probe dev %d <- dev %d: %.1f GB/s, %.1f us\n",
                          __func__, p->devices[d], p->devices[s],
                          bw_Bps[d][s] / 1e9, lat_us[d][s]);
        }
    }

    // A ring step is gated by its slowest edge, so minimize the bottleneck
    // edge time at a reference chunk size (tiebreak: total edge time).
    // Exact search over the (N-1)! distinct cycles for N <= 8 (position 0
    // fixed at device-list index 0 by rotation symmetry; reversal duplicates
    // are harmless); multi-start greedy + bounded 2-opt beyond that.
    const double ref_chunk = (double) (p->auto_ring_bytes / (size_t) n);
    auto edge_us = [&](int dst, int src) {
        return lat_us[dst][src] + ref_chunk / bw_Bps[dst][src] * 1e6;
    };

    int best[GGML_CUDA_MAX_DEVICES];
    for (int k = 0; k < n; ++k) {
        best[k] = k;
    }

    if (n <= 8) {
        int cand[GGML_CUDA_MAX_DEVICES];
        for (int k = 0; k < n; ++k) {
            cand[k] = k;
        }
        double best_max = 1e30, best_sum = 1e30;
        do {
            double cmax = 0, csum = 0;
            for (int pos = 0; pos < n; ++pos) {
                const double t = edge_us(cand[pos], cand[(pos - 1 + n) % n]);
                cmax = cmax < t ? t : cmax;
                csum += t;
            }
            if (cmax < best_max || (cmax == best_max && csum < best_sum)) {
                best_max = cmax;
                best_sum = csum;
                for (int k = 0; k < n; ++k) {
                    best[k] = cand[k];
                }
            }
        } while (std::next_permutation(cand + 1, cand + n));
    } else {
        // N > 8: (N-1)! is too large to enumerate. Multi-start nearest-neighbor
        // (one greedy pass per start node) polished by a bounded local search
        // combining 2-opt and Or-opt -- all cheap at N <= 16 and far better than
        // a single greedy pass. Edges are directional, so there is no cheap
        // incremental delta; each candidate recomputes the full cycle cost
        // (O(N), negligible at this N).
        auto cost = [&](const int * c, double * cmax, double * csum) {
            *cmax = 0; *csum = 0;
            for (int pos = 0; pos < n; ++pos) {
                const double t = edge_us(c[pos], c[(pos - 1 + n) % n]);
                *cmax = *cmax < t ? t : *cmax;
                *csum += t;
            }
        };
        auto better = [](double amax, double asum, double bmax, double bsum) {
            return amax < bmax || (amax == bmax && asum < bsum);
        };

        double best_max = 1e30, best_sum = 1e30;
        int    cand[GGML_CUDA_MAX_DEVICES];

        for (int start = 0; start < n; ++start) {
            bool used[GGML_CUDA_MAX_DEVICES] = {};
            cand[0]     = start;
            used[start] = true;
            for (int pos = 1; pos < n; ++pos) {
                int    bj = -1;
                double bt = 1e30;
                for (int j = 0; j < n; ++j) {
                    if (!used[j] && edge_us(j, cand[pos - 1]) < bt) {
                        bt = edge_us(j, cand[pos - 1]);
                        bj = j;
                    }
                }
                cand[pos] = bj;
                used[bj]  = true;
            }
            double cmax, csum;
            cost(cand, &cmax, &csum);
            if (better(cmax, csum, best_max, best_sum)) {
                best_max = cmax; best_sum = csum;
                for (int k = 0; k < n; ++k) best[k] = cand[k];
            }
        }

        // Local search: alternate a 2-opt sweep and an Or-opt sweep each pass,
        // up to a fixed cap; stop early on a pass with no improvement.
        for (int k = 0; k < n; ++k) cand[k] = best[k];
        for (int pass = 0; pass < 8; ++pass) {
            bool improved = false;

            // 2-opt: reverse cand[i+1..j] in place (cand stays == best at the
            // top of each trial, reverted on reject). Handles the directional
            // edges by recomputing full cost.
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    std::reverse(cand + i + 1, cand + j + 1);
                    double cmax, csum;
                    cost(cand, &cmax, &csum);
                    if (better(cmax, csum, best_max, best_sum)) {
                        best_max = cmax; best_sum = csum;
                        for (int k = 0; k < n; ++k) best[k] = cand[k];
                        improved = true;
                    } else {
                        std::reverse(cand + i + 1, cand + j + 1);  // revert
                    }
                }
            }

            // Or-opt: relocate a chain cand[i..i+L-1] (L in 1..3) to slot p
            // without reversing it -- reaches tours 2-opt cannot under
            // directional costs. Applying a move rebuilds cand, so rest[] would
            // go stale; apply the first improving move, then restart the sweep.
            bool or_moved = true;
            while (or_moved) {
                or_moved = false;
                const int Lmax = n - 1 < 3 ? n - 1 : 3;
                for (int L = 1; L <= Lmax && !or_moved; ++L) {
                    for (int i = 0; i + L <= n && !or_moved; ++i) {
                        int rest[GGML_CUDA_MAX_DEVICES], m = 0;
                        for (int k = 0; k < n; ++k) {
                            if (k < i || k >= i + L) rest[m++] = cand[k];
                        }
                        for (int pp = 0; pp <= m && !or_moved; ++pp) {
                            if (pp == i) continue;  // reinsert where it came from
                            int trial[GGML_CUDA_MAX_DEVICES], idx = 0;
                            for (int k = 0; k < pp; ++k)    trial[idx++] = rest[k];
                            for (int k = 0; k < L;  ++k)    trial[idx++] = cand[i + k];
                            for (int k = pp; k < m; ++k)    trial[idx++] = rest[k];
                            double cmax, csum;
                            cost(trial, &cmax, &csum);
                            if (better(cmax, csum, best_max, best_sum)) {
                                best_max = cmax; best_sum = csum;
                                for (int k = 0; k < n; ++k) best[k] = cand[k] = trial[k];
                                improved = true; or_moved = true;  // restart sweep
                            }
                        }
                    }
                }
            }

            if (!improved) break;
        }
    }
    for (int k = 0; k < n; ++k) {
        p->ring_order[k] = best[k];
    }

    double best_max = 0;
    for (int pos = 0; pos < n; ++pos) {
        const double t = edge_us(best[pos], best[(pos - 1 + n) % n]);
        best_max = best_max < t ? t : best_max;
    }
    char order_str[8 * GGML_CUDA_MAX_DEVICES + 1] = {};
    int len = 0;
    for (int k = 0; k < n; ++k) {
        len += snprintf(order_str + len, sizeof(order_str) - (size_t) len, "%s%d",
                        k == 0 ? "" : " -> ", p->devices[best[k]]);
    }
    GGML_LOG_INFO("%s: ring order %s (bottleneck edge %.1f us at %zu KiB chunk)\n",
                  __func__, order_str, best_max, (p->auto_ring_bytes / (size_t) n) >> 10);
}

// (Barrier events are managed inline in the run loop; see data_ready in
// the struct above.)

ggml_cuda_ar_pipeline_direct * ggml_cuda_ar_pipeline_direct_init(
        const int * devices, size_t n_devices) {

    if (n_devices < 2 || n_devices > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_DEBUG("%s: n_devices=%zu out of range [2,%d]; falling back\n",
                       __func__, n_devices, GGML_CUDA_MAX_DEVICES);
        return nullptr;
    }

    // Algorithm selection: GGML_CUDA_AR_DIRECT_ALGO=auto (default) |
    // butterfly | ring | bde | oneshot. Auto picks per call by tensor size (see
    // ggml_cuda_ar_allreduce_direct); oneshot is opt-in only until the bench box
    // has measured it.
    ggml_cuda_ar_algo algo = GGML_CUDA_AR_ALGO_AUTO;
    {
        const char * algo_env = getenv("GGML_CUDA_AR_DIRECT_ALGO");
        if (algo_env && algo_env[0] != '\0') {
            if (strcmp(algo_env, "ring") == 0) {
                algo = GGML_CUDA_AR_ALGO_RING;
            } else if (strcmp(algo_env, "butterfly") == 0) {
                algo = GGML_CUDA_AR_ALGO_BUTTERFLY;
            } else if (strcmp(algo_env, "bde") == 0) {
                algo = GGML_CUDA_AR_ALGO_BDE;
            } else if (strcmp(algo_env, "oneshot") == 0) {
                algo = GGML_CUDA_AR_ALGO_ONESHOT;
            } else if (strcmp(algo_env, "auto") != 0) {
                GGML_LOG_WARN("%s: unknown GGML_CUDA_AR_DIRECT_ALGO value '%s'; using auto\n",
                              __func__, algo_env);
            }
        }
    }

    // Power-of-2 only for butterfly's and bde's XOR pairing; ring is
    // chunk-based (reduce-scatter/all-gather) and works for any N >= 2.
    // Auto on a non-power-of-2 N simply always picks ring.
    if ((algo == GGML_CUDA_AR_ALGO_BUTTERFLY || algo == GGML_CUDA_AR_ALGO_BDE) &&
        (n_devices & (n_devices - 1)) != 0) {
        GGML_LOG_DEBUG("%s: n_devices=%zu not a power of 2; falling back to meta-butterfly\n",
                       __func__, n_devices);
        return nullptr;
    }

    // Probe full P2P reachability: every ordered pair must be reachable.
    // Bail (return nullptr) if any pair is not -- never make things worse.
    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, devices[i], devices[j]));
            if (!can) {
                GGML_LOG_DEBUG("%s: device %d cannot access peer %d; falling back\n",
                               __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }

    auto * p = new ggml_cuda_ar_pipeline_direct{};
    p->algo = algo;
    p->n_devices = (int) n_devices;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Initial per-device scratch size, NOT a cap -- anything larger grows on
    // demand (ggml_cuda_ar_ensure_tmp). Too low costs a few warmup growths, too
    // high wastes VRAM. Scratch is dev_tmp + dev_bf16 + dev_tmp_bf16 = 2x this
    // per device, times N devices, competing with KV cache.
    p->tmp_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DIRECT_TMP_BYTES", 16 * 1024 * 1024);
    if (p->tmp_bytes < 1024 * 1024) {
        p->tmp_bytes = 1024 * 1024;
    }

#if defined(GGML_HIP_AR_BF16)
    p->bf16_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DIRECT_BF16_THRESHOLD", 1 * 1024 * 1024);
#endif

    // algo=auto crossover, compared against the per-round working bytes (post-
    // bf16). Below it the AR is latency-bound and butterfly's log2(N) rounds
    // win; above it it is bandwidth-bound and bde's ~25% fewer bytes moved at
    // N=4 win. 1 MiB sits just past the measured butterfly/bde crossover
    // (~512-640 KiB at N=4).
    p->auto_ring_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DIRECT_AUTO_RING_BYTES", 1 * 1024 * 1024);

    // One-shot is a latency trick, so it is only chosen for small collectives.
    // 256 KiB is a guess pending the pp measurement; n_embd * ubatch * 4 B for
    // a 1024-token prefill ubatch is 10 MiB, well above anything decode sees.
    p->os_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DIRECT_ONESHOT_BYTES", 256 * 1024);

    // Enable peer access on every ordered pair. It is device-global state that
    // persists across the pipeline's lifetime, so a 2nd+ init finds it already
    // enabled: cudaErrorPeerAccessAlreadyEnabled is benign but leaves the
    // sticky flag set, so clear it explicitly. Any other error is a real
    // failure and falls back to the meta-butterfly.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(p->devices[j], 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) {
                // Benign: clear the sticky flag so later CUDA_CHECK is clean.
                (void) cudaGetLastError();
            } else if (e != cudaSuccess) {
                GGML_LOG_WARN("%s: cudaDeviceEnablePeerAccess(%d <- %d) failed: %s\n",
                              __func__, p->devices[i], p->devices[j], cudaGetErrorString(e));
                ggml_cuda_ar_pipeline_direct_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->data_ready[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for data_ready failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_direct_free(p);
            return nullptr;
        }

        if (cudaMalloc(&p->dev_tmp[i], p->tmp_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for tmp failed (%zu bytes) on device %d\n",
                           __func__, p->tmp_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_direct_free(p);
            return nullptr;
        }

        if (cudaMalloc(&p->os_inbox[i], ggml_cuda_ar_os_bytes(p, p->tmp_bytes)) != cudaSuccess ||
            cudaMemset(p->os_inbox[i], 0, ggml_cuda_ar_os_bytes(p, p->tmp_bytes)) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for the one-shot inbox failed (%zu bytes) on device %d\n",
                           __func__, ggml_cuda_ar_os_bytes(p, p->tmp_bytes), p->devices[i]);
            ggml_cuda_ar_pipeline_direct_free(p);
            return nullptr;
        }

#if defined(GGML_HIP_AR_BF16)
        if (cudaMalloc(&p->dev_bf16[i], p->tmp_bytes / 2) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for dev_bf16 failed (%zu bytes) on device %d\n",
                           __func__, p->tmp_bytes / 2, p->devices[i]);
            ggml_cuda_ar_pipeline_direct_free(p);
            return nullptr;
        }
        if (cudaMalloc(&p->dev_tmp_bf16[i], p->tmp_bytes / 2) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for dev_tmp_bf16 failed (%zu bytes) on device %d\n",
                           __func__, p->tmp_bytes / 2, p->devices[i]);
            ggml_cuda_ar_pipeline_direct_free(p);
            return nullptr;
        }
#endif
    }

    // Ring order: identity by default, measurement-based reordering when the
    // ring can run (algo=ring, or algo=auto with N > 2) or when N=4
    // butterfly/bde reuses the same probe for butterfly_order (below).
    for (int i = 0; i < p->n_devices; ++i) {
        p->ring_order[i] = i;
        p->butterfly_order[i] = i;
    }
    const bool want_bandwidth_probe =
        p->algo == GGML_CUDA_AR_ALGO_RING ||
        (p->algo == GGML_CUDA_AR_ALGO_AUTO && p->n_devices > 2) ||
        ((p->algo == GGML_CUDA_AR_ALGO_BUTTERFLY || p->algo == GGML_CUDA_AR_ALGO_BDE) &&
         p->n_devices == 4);
    if (want_bandwidth_probe) {
        ggml_cuda_ar_ring_pick_order(p);
    }

    // N=4 butterfly/bde device mapping: relabel ring_order onto butterfly's
    // fixed cycle sequence [0,1,3,2] (its 2-round XOR graph is isomorphic to a
    // 4-cycle). Only valid once the probe has run; identity otherwise.
    if (want_bandwidth_probe && p->n_devices == 4) {
        p->butterfly_order[0] = p->ring_order[0];
        p->butterfly_order[1] = p->ring_order[1];
        p->butterfly_order[2] = p->ring_order[3];
        p->butterfly_order[3] = p->ring_order[2];

        bool is_identity = true;
        for (int i = 0; i < 4; ++i) {
            is_identity = is_identity && p->butterfly_order[i] == i;
        }
        if (!is_identity) {
            GGML_LOG_INFO("%s: butterfly/bde device mapping (from ring probe): "
                          "L0=HIP%d L1=HIP%d L2=HIP%d L3=HIP%d\n",
                          __func__,
                          p->devices[p->butterfly_order[0]], p->devices[p->butterfly_order[1]],
                          p->devices[p->butterfly_order[2]], p->devices[p->butterfly_order[3]]);
        }
    }

    // Butterfly pairing (round r pairs device i with i XOR (1<<r)); bde reuses
    // the same table with half-range exchanges. Built whenever butterfly or bde
    // can be selected. n_rounds==0 means only ring can run.
    p->n_rounds = 0;
    if (p->algo == GGML_CUDA_AR_ALGO_BUTTERFLY || p->algo == GGML_CUDA_AR_ALGO_BDE ||
        ((p->algo == GGML_CUDA_AR_ALGO_AUTO || p->algo == GGML_CUDA_AR_ALGO_ONESHOT) &&
         (n_devices & (n_devices - 1)) == 0)) {
        size_t offset_j = n_devices / 2;
        while (offset_j >= 1) {
            const int r = p->n_rounds;
            int n_pairs = 0;
            for (size_t i = 0; i < n_devices; ++i) {
                size_t j = i ^ offset_j;
                if (j > i) {  // each pair listed once
                    p->pairs_in_round[r][2 * n_pairs + 0] = (int) i;
                    p->pairs_in_round[r][2 * n_pairs + 1] = (int) j;
                    n_pairs++;
                }
            }
            p->n_pairs_in_round[r] = n_pairs;
            p->n_rounds++;
            offset_j /= 2;
        }
    }

    ggml_cuda_ar_oneshot_probe(p);

    if (p->algo == GGML_CUDA_AR_ALGO_RING) {
        GGML_LOG_INFO("%s: initialized direct-P2P AllReduce pipeline: %zu GPUs, %zu KB tmp per GPU, "
                      "algo=ring (%d steps)\n",
                      __func__, n_devices, p->tmp_bytes >> 10, 2 * ((int) n_devices - 1));
    } else if (p->algo == GGML_CUDA_AR_ALGO_BDE) {
        GGML_LOG_INFO("%s: initialized direct-P2P AllReduce pipeline: %zu GPUs, %zu KB tmp per GPU, "
                      "algo=bde (%d rounds)\n",
                      __func__, n_devices, p->tmp_bytes >> 10, 2 * p->n_rounds);
    } else if (p->algo == GGML_CUDA_AR_ALGO_AUTO) {
        GGML_LOG_INFO("%s: initialized direct-P2P AllReduce pipeline: %zu GPUs, %zu KB tmp per GPU, "
                      "algo=auto (%s, bde for tensors >= %zu KB)\n",
                      __func__, n_devices, p->tmp_bytes >> 10,
                      p->n_rounds > 0 ? "butterfly below" : "ring only", p->auto_ring_bytes >> 10);
    } else if (p->algo == GGML_CUDA_AR_ALGO_ONESHOT) {
        GGML_LOG_INFO("%s: initialized direct-P2P AllReduce pipeline: %zu GPUs, %zu KB tmp per GPU, "
                      "algo=oneshot (1 launch per GPU per collective, %d x %d blocks)\n",
                      __func__, n_devices, p->tmp_bytes >> 10,
                      GGML_CUDA_AR_OS_BLOCKS, GGML_CUDA_AR_OS_THREADS);
    } else {
        GGML_LOG_INFO("%s: initialized direct-P2P AllReduce pipeline: %zu GPUs, %zu KB tmp per GPU, "
                      "algo=butterfly (%d rounds)\n",
                      __func__, n_devices, p->tmp_bytes >> 10, p->n_rounds);
    }
    return p;
}

void ggml_cuda_ar_pipeline_direct_free(ggml_cuda_ar_pipeline_direct * p) {
    if (!p) return;

    // No dedicated AR stream to drain; all AR work is on the callers'
    // compute streams, which the meta-backend synchronises around each AR.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->dev_tmp[i]) {
            (void)cudaFree(p->dev_tmp[i]);
        }
        if (p->os_inbox[i]) {
            (void)cudaFree(p->os_inbox[i]);
        }
#if defined(GGML_HIP_AR_BF16)
        if (p->dev_bf16[i]) {
            (void)cudaFree(p->dev_bf16[i]);
        }
        if (p->dev_tmp_bf16[i]) {
            (void)cudaFree(p->dev_tmp_bf16[i]);
        }
#endif
        if (p->data_ready[i]) {
            (void)cudaEventDestroy(p->data_ready[i]);
        }
    }
    delete p;
}

// Release every per-device scratch allocation. Shared by the growth path
// (which frees before reallocating larger) and its failure handler.
static void ggml_cuda_ar_free_tmp_buffers(ggml_cuda_ar_pipeline_direct * p) {
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->dev_tmp[i]) {
            (void) cudaFree(p->dev_tmp[i]);
            p->dev_tmp[i] = nullptr;
        }
        if (p->os_inbox[i]) {
            (void) cudaFree(p->os_inbox[i]);
            p->os_inbox[i] = nullptr;
        }
#if defined(GGML_HIP_AR_BF16)
        if (p->dev_bf16[i]) {
            (void) cudaFree(p->dev_bf16[i]);
            p->dev_bf16[i] = nullptr;
        }
        if (p->dev_tmp_bf16[i]) {
            (void) cudaFree(p->dev_tmp_bf16[i]);
            p->dev_tmp_bf16[i] = nullptr;
        }
#endif
    }
}

// Grow the per-device scratch so a tensor of `need_bytes` fits, returning
// false only if it cannot. High-water mark sized exactly to the largest tensor
// seen, never shrinks; growth only fires on a record-setting size, which is
// rare (payload is n_embd * n_tokens * type_size and n_tokens takes only a few
// values), so it settles after a few calls. Sizing per call would be the
// mistake -- cudaMalloc/cudaFree synchronise the device.
//
// Deliberately NOT ggml_cuda_pool: pool memory can be VMM-backed, which is only
// peer-accessible under GGML_CUDA_P2P / NCCL builds, but the round loop needs
// peers to read each other's dev_tmp -- plain cudaMalloc is peer-readable via
// the cudaDeviceEnablePeerAccess done at init. On failure the pipeline is
// disabled for the rest of the run (the meta-backend fallback needs comparable
// buffers, so anything that OOMs here would OOM there too).
static bool ggml_cuda_ar_ensure_tmp(ggml_cuda_ar_pipeline_direct * p, size_t need_bytes) {
    if (p->tmp_alloc_failed) {
        return false;
    }
    if (need_bytes <= p->tmp_bytes) {
        return true;
    }

    const size_t want = need_bytes;

    // Prior AR work reads these buffers across devices, so drain everything
    // before freeing to avoid a cross-device use-after-free.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Free before allocating so old+new are never resident at once.
    ggml_cuda_ar_free_tmp_buffers(p);

    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        bool ok = cudaMalloc(&p->dev_tmp[i], want) == cudaSuccess;
        // Reallocated with tmp, so the slot layout stays a pure function of
        // tmp_bytes (see ggml_cuda_ar_launch_oneshot). Zeroed because generation
        // 0 means never-written and random tags must not equal a real one.
        ok = ok && cudaMalloc(&p->os_inbox[i], ggml_cuda_ar_os_bytes(p, want)) == cudaSuccess;
        ok = ok && cudaMemset(p->os_inbox[i], 0, ggml_cuda_ar_os_bytes(p, want)) == cudaSuccess;
#if defined(GGML_HIP_AR_BF16)
        // bf16 buffers are half-width (2 bytes/elem vs F32's 4).
        ok = ok && cudaMalloc(&p->dev_bf16[i],     want / 2) == cudaSuccess;
        ok = ok && cudaMalloc(&p->dev_tmp_bf16[i], want / 2) == cudaSuccess;
#endif
        if (!ok) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: failed to grow direct-P2P AR scratch to %zu MiB on device %d; "
                           "disabling the direct pipeline for this run\n",
                           __func__, want >> 20, p->devices[i]);
            ggml_cuda_ar_free_tmp_buffers(p);
            p->tmp_alloc_failed = true;
            p->tmp_bytes        = 0;
            return false;
        }
    }

    GGML_LOG_INFO("%s: grew direct-P2P AR scratch %zu -> %zu MiB per GPU (tensor needed %zu MiB)\n",
                  __func__, p->tmp_bytes >> 20, want >> 20, need_bytes >> 20);
    p->tmp_bytes = want;
    return true;
}

// Chunk k's [offset, count) in elements, for splitting `ne` elements evenly
// across `n` chunks (first `ne % n` chunks get one extra element). Shared by
// both phases of the ring algorithm below.
static void ggml_cuda_ar_ring_chunk_bounds(int64_t ne, int n, int k, int64_t * offset, int64_t * count) {
    const int64_t base = ne / n;
    const int64_t rem  = ne % n;
    *count  = base + (k < rem ? 1 : 0);
    *offset = k * base + (k < rem ? k : rem);
}

// Dispatch the in-place add for one chunk, by work_type.
static void ggml_cuda_ar_ring_add_chunk(
        ggml_type work_type, void * dst, const void * src, int64_t count, cudaStream_t stream) {
    const int block_size = 256;
    int n_blocks = (int) ((count + block_size - 1) / block_size);
    if (n_blocks > 1024) n_blocks = 1024;
    if (n_blocks < 1)    n_blocks = 1;
    switch (work_type) {
        case GGML_TYPE_F32:
            ggml_cuda_ar_direct_add_kernel<float><<<n_blocks, block_size, 0, stream>>>(
                static_cast<float *>(dst), static_cast<const float *>(src), (int) count);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_ar_direct_add_kernel<half><<<n_blocks, block_size, 0, stream>>>(
                static_cast<half *>(dst), static_cast<const half *>(src), (int) count);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_ar_direct_add_kernel<nv_bfloat16><<<n_blocks, block_size, 0, stream>>>(
                static_cast<nv_bfloat16 *>(dst), static_cast<const nv_bfloat16 *>(src), (int) count);
            break;
        default: GGML_ASSERT(false);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Ring AllReduce (GGML_CUDA_AR_DIRECT_ALGO=ring): the device at ring position
// pos is p->ring_order[pos], its ring neighbors are positions (pos-1+N)%N and
// (pos+1)%N. Correct for any cyclic order (a permutation is just a relabeling);
// only speed changes. Chunk indices below are ring positions:
//   reduce-scatter: step s, position pos receives chunk ci=(pos-s-1+N)%N from
//     pos-1 and ADDS it. After N-1 steps position pos holds the full sum for
//     chunk (pos+1)%N.
//   all-gather: step s, position pos receives chunk ci=(pos-s+N)%N from pos-1
//     and OVERWRITES its own buffer (already fully reduced upstream).
//
// A single data_ready[i] per device (recorded after each step's write) is
// enough: a device's buffer is never simultaneously read-by-peer and
// written-by-self at the same chunk, and the within-step copy->add ordering is
// free via same-stream program order.
static void ggml_cuda_ar_allreduce_direct_ring(
        ggml_cuda_ar_pipeline_direct * p,
        ggml_backend_t              * backends,
        void                        ** work_data,
        void                        ** work_tmp,
        ggml_type                     work_type,
        int64_t                       ne) {
    const int n = p->n_devices;
    const size_t elem_size = ggml_type_size(work_type);

    auto stream_of = [&](int i) {
        return static_cast<ggml_backend_cuda_context *>(backends[i]->context)->stream();
    };

    auto do_step = [&](int s, bool reduce_scatter) {
        // Phase 1: each device copies its recv chunk from its ring-prev into
        // local tmp (cross-device, so must wait on prev's data_ready -- the
        // RAW hazard: prev's write of that exact chunk, from the previous
        // step or the initial entry barrier, must be visible first).
        for (int pos = 0; pos < n; ++pos) {
            const int i    = p->ring_order[pos];
            const int prev = p->ring_order[(pos - 1 + n) % n];
            const int ci = reduce_scatter ? (int) ((pos - s - 1 + n) % n) : (int) ((pos - s + n) % n);
            int64_t off, cnt;
            ggml_cuda_ar_ring_chunk_bounds(ne, n, ci, &off, &cnt);

            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = stream_of(i);
            CUDA_CHECK(cudaStreamWaitEvent(stream, p->data_ready[prev]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                work_tmp[i], p->devices[i],
                static_cast<char *>(work_data[prev]) + (size_t) off * elem_size, p->devices[prev],
                (size_t) cnt * elem_size, stream));
        }

        // Phase 2: consume tmp on the same stream that just filled it --
        // program order gives the read-after-write for free, no event
        // needed. Reduce-scatter adds; all-gather overwrites (copy).
        for (int pos = 0; pos < n; ++pos) {
            const int i  = p->ring_order[pos];
            const int ci = reduce_scatter ? (int) ((pos - s - 1 + n) % n) : (int) ((pos - s + n) % n);
            int64_t off, cnt;
            ggml_cuda_ar_ring_chunk_bounds(ne, n, ci, &off, &cnt);

            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = stream_of(i);
            void * dst = static_cast<char *>(work_data[i]) + (size_t) off * elem_size;
            if (reduce_scatter) {
                ggml_cuda_ar_ring_add_chunk(work_type, dst, work_tmp[i], cnt, stream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, work_tmp[i], (size_t) cnt * elem_size,
                                          cudaMemcpyDeviceToDevice, stream));
            }
        }

        // Step done: re-record data_ready so the next step's phase-1 copies
        // (or, on the last all-gather step, the caller) see this write.
        for (int i = 0; i < n; ++i) {
            CUDA_CHECK(cudaEventRecord(p->data_ready[i], stream_of(i)));
        }
    };

    for (int s = 0; s < n - 1; ++s) do_step(s, /*reduce_scatter=*/true);
    for (int s = 0; s < n - 1; ++s) do_step(s, /*reduce_scatter=*/false);
}

// Recursive-doubling butterfly AllReduce (GGML_CUDA_AR_DIRECT_ALGO=butterfly):
// round r pairs i with j = i XOR (1<<r); after log2(N) rounds every device
// holds the full sum. N = power of 2 (enforced at init). Two per-device buffers
// (work_data/work_tmp) swap roles each round in a ping-pong: round r reads
// src_buf=buf[r%2] (the running sum) and writes dst_buf=buf[1-(r%2)] (peer data
// copied in, then summed). For odd log2(N) (N=2, N=8) the result ends in
// work_tmp and is copied back to work_data at the end.
//
// The ping-pong makes phase-1 (read src_buf) and phase-2 (write dst_buf)
// disjoint within a round, so a single data_ready[i] carries the two cross-
// device dependencies, waited on twice per device per round:
//   RAW (this round's partner): phase-1 reads src_buf[peer], the partial sum
//     the peer produced last round. Wait on data_ready[partner_r].
//   WAR (last round's partner): dst_buf this round was src_buf last round, so
//     the device that read it from us (our round r-1 partner, not round r's)
//     must finish before phase-1 overwrites it. Wait on data_ready[partner_r-1];
//     round 0 needs none. Dropping this passes ordinary tests but is a genuine
//     race at N >= 4 with asymmetric per-pair P2P bandwidth.
static void ggml_cuda_ar_allreduce_direct_butterfly(
        ggml_cuda_ar_pipeline_direct * p,
        ggml_backend_t              * backends,
        void                        ** work_data,
        void                        ** work_tmp,
        ggml_type                     work_type,
        int64_t                       ne) {
    const int n = p->n_devices;
    const size_t elem_size = ggml_type_size(work_type);

    auto stream_of = [&](int i) {
        return static_cast<ggml_backend_cuda_context *>(backends[i]->context)->stream();
    };

    // Abstract-index partner of logical index i in round r (offsets n/2, ..., 1;
    // untranslated XOR space, translated through butterfly_order at use sites).
    auto partner_in_round = [&](int i, int r) {
        return i ^ ((n >> 1) >> r);
    };

    for (int r = 0; r < p->n_rounds; ++r) {
        const int parity = r & 1;
        void ** src_buf = parity ? work_tmp : work_data;
        void ** dst_buf = parity ? work_data : work_tmp;
        const size_t bytes = (size_t) ne * elem_size;
        const int n_pairs = p->n_pairs_in_round[r];

        // Phase 1: copies (concurrent across devices). Each device copies its
        // peer's source buffer into its own destination buffer. a_abs/b_abs are
        // the abstract XOR indices; a/b the actual resource indices.
        for (int k = 0; k < n_pairs; ++k) {
            const int a_abs = p->pairs_in_round[r][2 * k + 0];
            const int b_abs = p->pairs_in_round[r][2 * k + 1];
            const int a = p->butterfly_order[a_abs];
            const int b = p->butterfly_order[b_abs];

            // Cross-round WAR: wait for last round's partner (who read this
            // buffer when it was src_buf) before overwriting it. See header.
            if (r > 0) {
                const int prev_a = p->butterfly_order[partner_in_round(a_abs, r - 1)];
                const int prev_b = p->butterfly_order[partner_in_round(b_abs, r - 1)];

                ggml_cuda_set_device(p->devices[a]);
                CUDA_CHECK(cudaStreamWaitEvent(stream_of(a), p->data_ready[prev_a]));

                ggml_cuda_set_device(p->devices[b]);
                CUDA_CHECK(cudaStreamWaitEvent(stream_of(b), p->data_ready[prev_b]));
            }

            // cudaMemcpyPeerAsync (hipMemcpyPeerAsync on HIP): explicit cross-
            // device copy -- cudaMemcpyAsync(DeviceToDevice) does not reliably
            // traverse P2P for peer pointers on HIP. RAW: wait on the peer's
            // data_ready before reading its src_buf.
            ggml_cuda_set_device(p->devices[a]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(a), p->data_ready[b]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                dst_buf[a], p->devices[a], src_buf[b], p->devices[b],
                bytes, stream_of(a)));

            ggml_cuda_set_device(p->devices[b]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(b), p->data_ready[a]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                dst_buf[b], p->devices[b], src_buf[a], p->devices[a],
                bytes, stream_of(b)));
        }

        // No phase 1 -> phase 2 barrier: phase-1 read src_buf[*] and
        // phase-2 writes dst_buf[*] -- disjoint buffers, within the same
        // round. The intra-device copy-then-add on the same stream is
        // free via program order.

        // Phase 2: dst_buf[self] += src_buf[self] (the destination holds
        // peer_data just copied in; the source holds self's running sum;
        // sum them into the destination so the result lives in dst_buf).
        for (int k = 0; k < n_pairs; ++k) {
            const int a = p->butterfly_order[p->pairs_in_round[r][2 * k + 0]];
            const int b = p->butterfly_order[p->pairs_in_round[r][2 * k + 1]];

            ggml_cuda_set_device(p->devices[a]);
            ggml_cuda_ar_ring_add_chunk(work_type, dst_buf[a], src_buf[a], ne, stream_of(a));

            ggml_cuda_set_device(p->devices[b]);
            ggml_cuda_ar_ring_add_chunk(work_type, dst_buf[b], src_buf[b], ne, stream_of(b));
        }

        // Round done: dst_buf[*] now holds the round-r partial sum. Re-
        // record data_ready so the next round's phase-1 copies (which will
        // read dst_buf[*] as src_buf[*+1]) see it.
        for (int i = 0; i < n; ++i) {
            CUDA_CHECK(cudaEventRecord(p->data_ready[i], stream_of(i)));
        }
    }

    // For odd log2(N) (N=2, N=8, ...) the result is in work_tmp; copy it back
    // to work_data so the caller's tensor holds the reduced value in place.
    if (p->n_rounds & 1) {
        const size_t bytes = (size_t) ne * elem_size;
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaMemcpyAsync(work_data[i], work_tmp[i], bytes,
                                       cudaMemcpyDeviceToDevice, stream_of(i)));
        }
    }
}

// Active range [lo, hi) of device i's buffer after `upto_round` rounds
// (0-indexed) of bde's recursive-halving reduce-scatter; upto_round == -1 gives
// the initial full range [0, ne). Pure function -- both phases call it to
// recover the range at any round. Mirrors pairs_in_round's offset schedule
// (n/2, ..., 1): each level bisects [lo, hi) (low half gets the odd element)
// and device i descends into the low half if its bit at that offset is 0. Since
// pairs are (a, b) with a < b = a ^ offset, a is always the low side.
static void ggml_cuda_ar_bde_active_range(
        int64_t ne, int n, int i, int upto_round, int64_t * lo, int64_t * hi) {
    int64_t l = 0, h = ne;
    int offset = n / 2;
    for (int r = 0; r <= upto_round; ++r) {
        const int64_t mid = l + (h - l + 1) / 2;
        if ((i & offset) == 0) {
            h = mid;
        } else {
            l = mid;
        }
        offset /= 2;
    }
    *lo = l;
    *hi = h;
}

// Rabenseifner's algorithm (GGML_CUDA_AR_DIRECT_ALGO=bde): recursive-halving
// reduce-scatter then recursive-doubling all-gather, both over the same XOR
// network as butterfly. N = power of 2 (enforced at init).
//
// Reduce-scatter, round r: for pair (a, b) with a < b, both hold the full
// sum-so-far over their shared range [lo, hi) (active_range at r-1). a keeps
// the low half [lo, mid), b the high half [mid, hi); each sends the half it
// gives up and adds in the half it receives. After n_rounds device i's range is
// its 1/N-sized final chunk. All-gather replays the pairs in reverse round
// order, extending each range back out via overwrites (no adds).
//
// Only ONE barrier per round (data_ready): a's phase-2 add writes [lo, mid),
// disjoint from the [mid, hi) its peer read in phase-1, so there is no RAW on
// either device's own memory (unlike butterfly's full-tensor exchange). Local
// copy-then-add ordering is free via same-stream program order.
static void ggml_cuda_ar_allreduce_direct_bde(
        ggml_cuda_ar_pipeline_direct * p,
        ggml_backend_t              * backends,
        void                        ** work_data,
        void                        ** work_tmp,
        ggml_type                     work_type,
        int64_t                       ne) {
    const int n = p->n_devices;
    const size_t elem_size = ggml_type_size(work_type);

    auto stream_of = [&](int i) {
        return static_cast<ggml_backend_cuda_context *>(backends[i]->context)->stream();
    };
    auto byte_ptr = [&](void * base, int64_t off) {
        return static_cast<char *>(base) + (size_t) off * elem_size;
    };

    auto record_data_ready = [&]() {
        for (int i = 0; i < n; ++i) {
            CUDA_CHECK(cudaEventRecord(p->data_ready[i], stream_of(i)));
        }
    };

    // --- Reduce-scatter: rounds 0 .. n_rounds-1 -----------------------------
    for (int r = 0; r < p->n_rounds; ++r) {
        const int n_pairs = p->n_pairs_in_round[r];

        // Phase 1: mutual half-range copies (concurrent across pairs). a/b are
        // the abstract XOR indices (active_range needs them untranslated);
        // a_phys/b_phys are the actual resource indices used for every access.
        for (int k = 0; k < n_pairs; ++k) {
            const int a = p->pairs_in_round[r][2 * k + 0];
            const int b = p->pairs_in_round[r][2 * k + 1];
            const int a_phys = p->butterfly_order[a];
            const int b_phys = p->butterfly_order[b];

            int64_t lo, hi;
            ggml_cuda_ar_bde_active_range(ne, n, a, r - 1, &lo, &hi);
            const int64_t mid    = lo + (hi - lo + 1) / 2;
            const int64_t cnt_lo = mid - lo;
            const int64_t cnt_hi = hi - mid;

            // a receives b's low half [lo, mid) -- a will add it into its
            // own low half, which it is keeping.
            ggml_cuda_set_device(p->devices[a_phys]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(a_phys), p->data_ready[b_phys]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                work_tmp[a_phys], p->devices[a_phys], byte_ptr(work_data[b_phys], lo), p->devices[b_phys],
                (size_t) cnt_lo * elem_size, stream_of(a_phys)));

            // b receives a's high half [mid, hi) -- b will add it into its
            // own high half, which it is keeping.
            ggml_cuda_set_device(p->devices[b_phys]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(b_phys), p->data_ready[a_phys]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                work_tmp[b_phys], p->devices[b_phys], byte_ptr(work_data[a_phys], mid), p->devices[a_phys],
                (size_t) cnt_hi * elem_size, stream_of(b_phys)));
        }

        // No phase1->phase2 barrier needed here, unlike butterfly: each
        // device's phase-2 add writes only the half it is keeping (a's
        // [lo, mid) / b's [mid, hi)), which is disjoint from the half its
        // peer just read from it (b read a's [mid, hi) / a read b's
        // [lo, mid)) -- so there is no read-after-write hazard on either
        // device's own memory to guard against, and the local
        // copy-then-add ordering is already free via same-stream program
        // order (same argument as the ring all-gather below).

        // Phase 2: adds, in place, over the half each device is keeping.
        for (int k = 0; k < n_pairs; ++k) {
            const int a = p->pairs_in_round[r][2 * k + 0];
            const int b = p->pairs_in_round[r][2 * k + 1];
            const int a_phys = p->butterfly_order[a];
            const int b_phys = p->butterfly_order[b];

            int64_t lo, hi;
            ggml_cuda_ar_bde_active_range(ne, n, a, r - 1, &lo, &hi);
            const int64_t mid    = lo + (hi - lo + 1) / 2;
            const int64_t cnt_lo = mid - lo;
            const int64_t cnt_hi = hi - mid;

            ggml_cuda_set_device(p->devices[a_phys]);
            ggml_cuda_ar_ring_add_chunk(work_type, byte_ptr(work_data[a_phys], lo), work_tmp[a_phys],
                                        cnt_lo, stream_of(a_phys));

            ggml_cuda_set_device(p->devices[b_phys]);
            ggml_cuda_ar_ring_add_chunk(work_type, byte_ptr(work_data[b_phys], mid), work_tmp[b_phys],
                                        cnt_hi, stream_of(b_phys));
        }

        // Round done: re-record data_ready so the next round's phase-1
        // copies (or, at the last round, the all-gather phase below) see it.
        record_data_ready();
    }

    // --- All-gather: rounds n_rounds-1 .. 0, mirroring reduce-scatter -------
    for (int r = p->n_rounds - 1; r >= 0; --r) {
        const int n_pairs = p->n_pairs_in_round[r];

        // Phase 1: mutual half-range copies of already-final data.
        for (int k = 0; k < n_pairs; ++k) {
            const int a = p->pairs_in_round[r][2 * k + 0];
            const int b = p->pairs_in_round[r][2 * k + 1];
            const int a_phys = p->butterfly_order[a];
            const int b_phys = p->butterfly_order[b];

            int64_t lo, hi;
            ggml_cuda_ar_bde_active_range(ne, n, a, r - 1, &lo, &hi);
            const int64_t mid    = lo + (hi - lo + 1) / 2;
            const int64_t cnt_lo = mid - lo;
            const int64_t cnt_hi = hi - mid;

            // a receives b's high half [mid, hi) (final, from b's deeper
            // rounds) to extend its own range back out to [lo, hi).
            ggml_cuda_set_device(p->devices[a_phys]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(a_phys), p->data_ready[b_phys]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                work_tmp[a_phys], p->devices[a_phys], byte_ptr(work_data[b_phys], mid), p->devices[b_phys],
                (size_t) cnt_hi * elem_size, stream_of(a_phys)));

            // b receives a's low half [lo, mid) to extend its own range.
            ggml_cuda_set_device(p->devices[b_phys]);
            CUDA_CHECK(cudaStreamWaitEvent(stream_of(b_phys), p->data_ready[a_phys]));
            CUDA_CHECK(cudaMemcpyPeerAsync(
                work_tmp[b_phys], p->devices[b_phys], byte_ptr(work_data[a_phys], lo), p->devices[a_phys],
                (size_t) cnt_lo * elem_size, stream_of(b_phys)));
        }

        // Phase 2: overwrite the disjoint (not-yet-owned) half from tmp.
        // Same-stream program order gives the phase1->phase2 RAW for free
        // (see the ring all-gather comment); no copy_done needed here.
        for (int k = 0; k < n_pairs; ++k) {
            const int a = p->pairs_in_round[r][2 * k + 0];
            const int b = p->pairs_in_round[r][2 * k + 1];
            const int a_phys = p->butterfly_order[a];
            const int b_phys = p->butterfly_order[b];

            int64_t lo, hi;
            ggml_cuda_ar_bde_active_range(ne, n, a, r - 1, &lo, &hi);
            const int64_t mid    = lo + (hi - lo + 1) / 2;
            const int64_t cnt_lo = mid - lo;
            const int64_t cnt_hi = hi - mid;

            ggml_cuda_set_device(p->devices[a_phys]);
            CUDA_CHECK(cudaMemcpyAsync(byte_ptr(work_data[a_phys], mid), work_tmp[a_phys],
                                      (size_t) cnt_hi * elem_size,
                                      cudaMemcpyDeviceToDevice, stream_of(a_phys)));

            ggml_cuda_set_device(p->devices[b_phys]);
            CUDA_CHECK(cudaMemcpyAsync(byte_ptr(work_data[b_phys], lo), work_tmp[b_phys],
                                      (size_t) cnt_lo * elem_size,
                                      cudaMemcpyDeviceToDevice, stream_of(b_phys)));
        }

        // Round done: re-record data_ready so the next (shallower) all-gather
        // round, or the caller once r==0 finishes, sees the wider range.
        record_data_ready();
    }
}

bool ggml_cuda_ar_allreduce_direct(
        ggml_cuda_ar_pipeline_direct * p,
        ggml_backend_t              * backends,
        ggml_tensor                ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    const ggml_type t = tensors[0]->type;
    GGML_ASSERT(t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t type_size = ggml_type_size(t);
    const size_t nbytes    = (size_t) ne * type_size;
    // Scratch must hold the whole tensor (no outer chunker); grow to fit rather
    // than fall back. Only an allocation failure returns false here.
    if (!ggml_cuda_ar_ensure_tmp(p, nbytes)) {
        return false;
    }

    // Selected up front: the one-shot path neither zeroes inactive shards nor
    // records data_ready events, so the choice must be known before both.
    //
    // algo=oneshot is still a per-call decision above os_bytes: one-shot moves
    // (n-1) wire-doubled copies of the payload instead of ring's 2*(n-1)/N
    // chunks, so at prefill sizes its bytes cost far more than the rounds it
    // saves, and the bf16 compression that would offset it is a build flag off
    // by default. Above the cutoff this falls back to what auto would pick.
    const size_t work_bytes = (size_t) ne * ggml_type_size(t);
    ggml_cuda_ar_algo algo = p->algo;

    if (algo == GGML_CUDA_AR_ALGO_AUTO ||
        (algo == GGML_CUDA_AR_ALGO_ONESHOT && work_bytes > p->os_bytes)) {
        const bool latency_bound = n == 2 || work_bytes < p->auto_ring_bytes;
        algo = p->n_rounds == 0 ? GGML_CUDA_AR_ALGO_RING
             : latency_bound    ? GGML_CUDA_AR_ALGO_BUTTERFLY
                                : GGML_CUDA_AR_ALGO_BDE;
    }
    const bool oneshot = algo == GGML_CUDA_AR_ALGO_ONESHOT;

    // Match NCCL semantics: inactive shards contribute zeros. The one-shot
    // pushes zeros for them instead of paying a memset per shard.
    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }
    for (int i = 0; i < n && !oneshot; ++i) {
        if (!compute_flag[i]) {
            ggml_cuda_set_device(p->devices[i]);
            auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, nbytes, ctx->stream()));
        }
    }

    // GGML_HIP_AR_BF16: the round loops operate on work_data/work_tmp, so the
    // code is identical either way. F32 tensors at/above the threshold compress
    // to the bf16 scratch (half the bytes moved); F16/BF16 (already 2
    // bytes/elem) always take the uncompressed path.
    bool use_bf16_compress = false;
#if defined(GGML_HIP_AR_BF16)
    use_bf16_compress = t == GGML_TYPE_F32 && nbytes >= p->bf16_threshold && !oneshot;
#endif
    const ggml_type work_type  = use_bf16_compress ? GGML_TYPE_BF16 : t;

    void * work_data[GGML_CUDA_MAX_DEVICES];
    void * work_tmp [GGML_CUDA_MAX_DEVICES];
    for (int i = 0; i < n; ++i) {
#if defined(GGML_HIP_AR_BF16)
        work_data[i] = use_bf16_compress ? p->dev_bf16[i]     : tensors[i]->data;
        work_tmp [i] = use_bf16_compress ? p->dev_tmp_bf16[i] : p->dev_tmp[i];
#else
        work_data[i] = tensors[i]->data;
        work_tmp [i] = p->dev_tmp[i];
#endif
    }

#if defined(GGML_HIP_AR_BF16)
    if (use_bf16_compress) {
        static const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            to_bf16(tensors[i]->data, static_cast<nv_bfloat16 *>(work_data[i]), ne, ctx->stream());
            CUDA_CHECK(cudaGetLastError());
        }
    }
#endif

    // Record each device's compute-stream progress as data_ready: the meta-
    // backend queues each device's subgraph on its compute stream and calls us
    // immediately, so the per-device compute (and the bf16-compress kernel
    // above) is still in flight -- peers must wait on this before reading in
    // round 0, or they would read stale data. The one-shot reads only its own
    // inbox, which is written by peers' pushes, so it needs no such handshake.
    if (!oneshot) {
        for (int i = 0; i < n; ++i) {
            auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            CUDA_CHECK(cudaEventRecord(p->data_ready[i], ctx->stream()));
        }
    }

    if (oneshot) {
        for (int i = 0; i < n; ++i) {
            if (p->os_inbox[i] == nullptr) {
                GGML_LOG_WARN("%s: one-shot inbox missing on device %d; using the meta fallback\n",
                              __func__, p->devices[i]);
                return false;
            }
        }

        // One 8 B wire unit carries 4 B of payload, so the tensor has to be a
        // multiple of 4 bytes; every AR tensor here is n_embd wide.
        if ((nbytes & 3) != 0) {
            GGML_LOG_DEBUG("%s: one-shot needs a 4-byte multiple, got %zu; falling back\n",
                           __func__, nbytes);
            return false;
        }

        cudaStream_t streams[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            streams[i] = ctx->stream();
        }

        ggml_cuda_ar_launch_oneshot(p, work_data, work_type, ne, compute_flag, streams);
        return true;
    }

    if (algo == GGML_CUDA_AR_ALGO_RING) {
        ggml_cuda_ar_allreduce_direct_ring(p, backends, work_data, work_tmp, work_type, ne);
    } else if (algo == GGML_CUDA_AR_ALGO_BDE) {
        ggml_cuda_ar_allreduce_direct_bde(p, backends, work_data, work_tmp, work_type, ne);
    } else {
        ggml_cuda_ar_allreduce_direct_butterfly(p, backends, work_data, work_tmp, work_type, ne);
    }

#if defined(GGML_HIP_AR_BF16)
    if (use_bf16_compress) {
        static const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            to_fp32(work_data[i], static_cast<float *>(tensors[i]->data), ne, ctx->stream());
            CUDA_CHECK(cudaGetLastError());
        }
    }
#endif

    return true;
}

#endif // defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)

#if !defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)

// Neither CUDA nor HIP runtime. The direct-P2P butterfly is unavailable;
// the dispatcher in allreduce.cu treats a nullptr pipeline as "init failed"
// and falls through to the next candidate implementation.
ggml_cuda_ar_pipeline_direct * ggml_cuda_ar_pipeline_direct_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_direct_free(ggml_cuda_ar_pipeline_direct *) {
}
bool ggml_cuda_ar_allreduce_direct(ggml_cuda_ar_pipeline_direct *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
