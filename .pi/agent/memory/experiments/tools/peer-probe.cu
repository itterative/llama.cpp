// Standalone HIP probe: can a kernel touch a peer GPU's memory over this box's
// PCIe root complex at all, and how long does it take. Written to debug the
// one-shot allreduce (E0xx plan), which is the first code in llama.cpp to
// dereference a peer pointer inside a kernel -- every existing path in
// allreduce-p2p.cu moves data with cudaMemcpyPeerAsync, which goes through the
// copy engine and does not need kernel-visible peer mappings.
//
// Build and run on the bench box:
//   /opt/rocm/bin/hipcc -O2 -o peer-probe .pi/agent/memory/experiments/tools/peer-probe.cu
//   ./peer-probe              # devices 0 and 1
//   ./peer-probe 0 2
//
// Deliberately hang-free: tests 1 and 2 do no cross-device waiting at all, and
// test 3's spin is bounded and reports what it saw instead of wedging.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                \
    do {                                                                        \
        hipError_t e_ = (x);                                                    \
        if (e_ != hipSuccess) {                                                 \
            printf("FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x,            \
                   hipGetErrorString(e_));                                      \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// 1: write to a peer pointer, no fence.
__global__ void push_plain(unsigned long long * dst, unsigned long long v) {
    *dst = v;
}

// 2: same, fenced. Whether this matters and whether 1 works at all are the two
// questions the one-shot design depends on.
__global__ void push_fenced(unsigned long long * dst, unsigned long long v) {
    *dst = v;
    __threadfence_system();
}

// 3: read a peer pointer into local memory.
__global__ void pull(const unsigned * src, unsigned * dst) {
    *dst = *src;
}

// Destination-side kernel copying its own local word somewhere the host can read.
// Peer stores show up here even when hipMemcpy of the same address reads back
// zero, so this is the only way to ask whether the store landed where a kernel on
// the destination can see it -- the property the allreduce actually needs.
__global__ void relay_local(const unsigned long long * src, unsigned long long * dst) {
    *dst = *src;
}

// 4: LL round trip with a bounded spin: write one tagged unit to the peer, then
// wait for the peer's unit in my own inbox. Reports the tag it actually saw.
__global__ void ll_round_trip(unsigned long long * to_peer, const unsigned long long * from_peer,
                              unsigned gen, unsigned * outcome, long cap) {
    *to_peer = ((unsigned long long) 0xA5A5A5A5u << 32) | gen;
    __threadfence_system();

    long spins = 0;
    unsigned long long w;
    const volatile unsigned long long * slot = from_peer;

    do {
        w = *slot;
        if ((unsigned) w == gen) {
            break;
        }
    } while (++spins < cap);

    *outcome = ((unsigned) w == gen) ? (unsigned) spins : 0xFFFFFFFFu;
}

int main(int argc, char ** argv) {
    const int da = argc > 1 ? atoi(argv[1]) : 0;
    const int db = argc > 2 ? atoi(argv[2]) : 1;

    int can_ab = 0;
    int can_ba = 0;
    CHECK(hipSetDevice(da));
    CHECK(hipDeviceCanAccessPeer(&can_ab, da, db));
    CHECK(hipSetDevice(db));
    CHECK(hipDeviceCanAccessPeer(&can_ba, db, da));
    printf("canAccessPeer %d->%d = %d, %d->%d = %d\n", da, db, can_ab, db, da, can_ba);

    CHECK(hipSetDevice(da));
    hipError_t ea = hipDeviceEnablePeerAccess(db, 0);
    CHECK(hipSetDevice(db));
    hipError_t eb = hipDeviceEnablePeerAccess(da, 0);
    printf("EnablePeerAccess: %s / %s\n", hipGetErrorString(ea), hipGetErrorString(eb));

    unsigned long long * buf_a;
    unsigned long long * buf_b;
    CHECK(hipSetDevice(da));
    CHECK(hipMalloc(&buf_a, 4096));
    CHECK(hipMemset(buf_a, 0, 4096));
    CHECK(hipSetDevice(db));
    CHECK(hipMalloc(&buf_b, 4096));
    CHECK(hipMemset(buf_b, 0, 4096));

    unsigned * flag_a;
    unsigned * flag_b;
    CHECK(hipSetDevice(da));
    CHECK(hipMalloc(&flag_a, sizeof(unsigned)));
    CHECK(hipSetDevice(db));
    CHECK(hipMalloc(&flag_b, sizeof(unsigned)));

    const unsigned long long pat = 0x1122334455667788ull;
    unsigned long long host = 0;

    unsigned long long * relay = nullptr;   // on device b, written by a b-local kernel
    CHECK(hipSetDevice(db));
    CHECK(hipMalloc(&relay, sizeof(unsigned long long)));
    CHECK(hipMemset(relay, 0xFF, sizeof(unsigned long long)));

    // Tests 1 and 2: one remote store, checked two ways. The DMA read is expected
    // to miss even when the store landed, since a peer write surfaces in the
    // destination's cache hierarchy while the copy engine reads DRAM. Distinct
    // patterns per variant so a stale line shows up as the other value.
    struct variant { const char * name; unsigned long long pat; bool fence; };
    const variant variants[2] = {
        { "test1 remote store, no fence ", 0x1111111122222222ull, false },
        { "test2 remote store, sys fence", 0x3333333344444444ull, true  },
    };

    for (int t = 0; t < 2; ++t) {
        const variant & v = variants[t];

        CHECK(hipSetDevice(da));
        if (v.fence) {
            push_fenced<<<1, 1>>>(buf_b, v.pat);
        } else {
            push_plain<<<1, 1>>>(buf_b, v.pat);
        }
        printf("%s launch: %s\n", v.name, hipGetErrorString(hipGetLastError()));
        CHECK(hipDeviceSynchronize());

        CHECK(hipSetDevice(db));
        relay_local<<<1, 1>>>(buf_b, relay);
        CHECK(hipDeviceSynchronize());

        unsigned long long seen = 0;
        CHECK(hipMemcpy(&seen, relay, sizeof(seen), hipMemcpyDeviceToHost));

        unsigned long long dma = 0;
        CHECK(hipMemcpy(&dma, buf_b, sizeof(dma), hipMemcpyDeviceToHost));

        printf("%-33s kernel-on-destination: %-9s (%016llx)  hipMemcpy: %s\n", v.name,
               seen == v.pat ? "LANDED" :
               (seen == 0xFFFFFFFFFFFFFFFFull ? "UNTOUCHED" :
               (seen == variants[1-t].pat ? "STALE-OLD" : "WRONG")),
               seen, dma == v.pat ? "sees it" : "stale (expected)");
    }

    // test 3: remote read
    CHECK(hipSetDevice(db));
    CHECK(hipMemset(buf_b, 0, 4096));
    host = 0xCAFEBABEull;
    CHECK(hipMemcpy(buf_b, &host, sizeof(host), hipMemcpyHostToDevice));
    CHECK(hipSetDevice(da));
    pull<<<1, 1>>>((const unsigned *) buf_b, flag_a);
    printf("test3 launch: %s\n", hipGetErrorString(hipGetLastError()));
    CHECK(hipDeviceSynchronize());
    unsigned low = 0;
    CHECK(hipMemcpy(&low, flag_a, sizeof(low), hipMemcpyDeviceToHost));
    printf("test3 remote read:                 %s (got 0x%08x, want the low word of 0xCAFEBABE)\n",
           low == 0xCAFEBABEu ? "WORKS" : "WRONG", low);

    // test 4: LL round trip. Both sides must be launched before either waits,
    // exactly as the real path does from one host thread.
    const long cap = 2000000;
    CHECK(hipSetDevice(da));
    ll_round_trip<<<1, 1>>>(buf_b, buf_a, 1, flag_a, cap);
    CHECK(hipSetDevice(db));
    ll_round_trip<<<1, 1>>>(buf_a, buf_b, 1, flag_b, cap);

    auto t0 = std::chrono::steady_clock::now();
    CHECK(hipSetDevice(da));
    CHECK(hipDeviceSynchronize());
    CHECK(hipSetDevice(db));
    CHECK(hipDeviceSynchronize());
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

    unsigned oa = 0;
    unsigned ob = 0;
    CHECK(hipSetDevice(da));
    CHECK(hipMemcpy(&oa, flag_a, sizeof(oa), hipMemcpyDeviceToHost));
    CHECK(hipSetDevice(db));
    CHECK(hipMemcpy(&ob, flag_b, sizeof(ob), hipMemcpyDeviceToHost));

    printf("test4 LL round trip: A saw %s (%u spins), B saw %s (%u spins), %.1f us\n",
           oa == 0xFFFFFFFFu ? "TIMEOUT" : "MATCH", oa,
           ob == 0xFFFFFFFFu ? "TIMEOUT" : "MATCH", ob, us);

    // test 5: cost of N sequential remote writes, as a per-op lower bound.
    CHECK(hipSetDevice(da));
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i) {
        push_fenced<<<1, 1>>>(buf_b, pat + i);
    }
    CHECK(hipDeviceSynchronize());
    printf("test5 200 remote writes: %.2f us per launch+store\n",
           std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200.0);

    printf("\nreading the lines above:\n"
           "  kernel-on-destination LANDED -> the inbox primitive works; that is all the AR needs\n"
           "  hipMemcpy stale              -> normal for peer stores, the AR never DMA-reads an inbox\n"
           "  test4 MATCH                  -> the LL tag round trip completes with a bounded spin\n"
           "  test5                        -> host floor per collective is n_devices x that number\n");
    return 0;
}
