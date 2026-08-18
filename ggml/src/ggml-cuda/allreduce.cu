#include "allreduce.cuh"
#include "allreduce-host.cuh"
#include "allreduce-p2p.cuh"

#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// AllReduce pipeline dispatcher.
//
// Thin wrapper: init tries each implementation in order and keeps the first
// that succeeds; exactly one of p->host_staged / p->direct ends up non-null,
// and that pointer is what free/allreduce dispatch on. An init returning
// nullptr means "not usable on this topology" (host-staged needs exactly 2
// CUDA devices; direct-P2P needs peer access on every pair), not an error, so
// init falls through to the next candidate. The caller in ggml-cuda.cu falls
// back to the meta-backend's generic AllReduce only if every candidate here
// returns nullptr.
//
// Order: 2-GPU host-staged first (tuned pinned-staging path for the no-P2P
// case), then N-GPU direct-P2P. GGML_CUDA_AR_PIPELINE=host_staged|direct forces
// a single candidate (still nullptr if it can't support the topology).
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {
    if (n_devices == 0 || n_devices > GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }

    const char * pipeline_env = getenv("GGML_CUDA_AR_PIPELINE");
    const bool   try_direct      = !pipeline_env || strcmp(pipeline_env, "host_staged") != 0;
    const bool   try_host_staged = !pipeline_env || strcmp(pipeline_env, "direct")      != 0;

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices = (int) n_devices;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    if (try_host_staged) {
        p->host_staged = ggml_cuda_ar_pipeline_init_host_staged(devices, n_devices);

        if (p->host_staged) {
            return p;
        }

        // clear sticky error from the failed host-staged init
        (void) cudaGetLastError();
    }

    if (try_direct) {
        p->direct = ggml_cuda_ar_pipeline_direct_init(devices, n_devices);

        if (p->direct) {
            return p;
        }

        // clear sticky error from the failed direct init
        (void) cudaGetLastError();
    }

    // cleanup, can't use either
    delete p;
    return nullptr;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Both frees are no-ops on nullptr, so no branch is needed here.
    ggml_cuda_ar_pipeline_direct_free(p->direct);
    ggml_cuda_ar_pipeline_free_host_staged(p->host_staged);

    delete p;
}

bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline * p, ggml_backend_t * backends, ggml_tensor ** tensors) {
    GGML_ASSERT(p != nullptr);

    if (p->host_staged) {
        return ggml_cuda_ar_allreduce_host_staged(p->host_staged, backends, tensors);
    }
    if (p->direct) {
        return ggml_cuda_ar_allreduce_direct(p->direct, backends, tensors);
    }

    GGML_ABORT("ggml_cuda_ar_pipeline has neither host_staged nor direct set");
}
