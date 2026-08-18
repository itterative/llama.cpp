#pragma once

// Private interface between allreduce.cu (dispatcher) and allreduce-host.cu
// (2-GPU host-staged pipeline). Not included by anything outside this pair.

#include "allreduce.cuh" // forward-declares ggml_cuda_ar_pipeline_host_staged; defined in allreduce-host.cu

ggml_cuda_ar_pipeline_host_staged * ggml_cuda_ar_pipeline_init_host_staged(
    const int * devices, size_t n_devices);

void ggml_cuda_ar_pipeline_free_host_staged(ggml_cuda_ar_pipeline_host_staged * pipeline);

bool ggml_cuda_ar_allreduce_host_staged(
    ggml_cuda_ar_pipeline_host_staged * pipeline,
    ggml_backend_t                    * backends,
    ggml_tensor                      ** tensors);
