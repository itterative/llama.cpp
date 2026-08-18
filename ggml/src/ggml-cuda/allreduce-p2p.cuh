#pragma once

// Private interface between allreduce.cu (dispatcher) and allreduce-p2p.cu
// (N-GPU direct-P2P butterfly/ring/bde pipeline). Not included by anything
// outside this pair.
//
// Internal names keep the "_direct" suffix from the pipeline's origin (the
// local rocm-optimizations branch's "direct-P2P" AllReduce) rather than
// "_p2p" -- so this file stays a near-verbatim port. The `direct` field name
// in ggml_cuda_ar_pipeline (allreduce.cuh) is what callers actually key off.

#include "allreduce.cuh" // forward-declares ggml_cuda_ar_pipeline_direct; defined in allreduce-p2p.cu

ggml_cuda_ar_pipeline_direct * ggml_cuda_ar_pipeline_direct_init(
    const int * devices, size_t n_devices);

void ggml_cuda_ar_pipeline_direct_free(ggml_cuda_ar_pipeline_direct * pipeline);

bool ggml_cuda_ar_allreduce_direct(
    ggml_cuda_ar_pipeline_direct * pipeline,
    ggml_backend_t               * backends,
    ggml_tensor                 ** tensors);
