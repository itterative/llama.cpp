#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque; each is defined in its own TU (allreduce-host.cu / allreduce-p2p.cu)
// and never included here, so this header stays a forward-declaration-only
// dependency for both.
struct ggml_cuda_ar_pipeline_host_staged;
struct ggml_cuda_ar_pipeline_direct;

// Thin dispatcher struct: n_devices/devices are duplicated from the backend-
// specific impl for cheap access without a downcast. Exactly one of
// host_staged/direct is non-null -- that pointer *is* the tag, so init/free/
// allreduce in allreduce.cu dispatch on which one is set rather than keeping
// a separate enum that could drift out of sync with it.
struct ggml_cuda_ar_pipeline {
    int                             n_devices;
    int                             devices[GGML_CUDA_MAX_DEVICES];
    ggml_cuda_ar_pipeline_host_staged * host_staged = nullptr;
    ggml_cuda_ar_pipeline_direct      * direct      = nullptr;
};

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);
