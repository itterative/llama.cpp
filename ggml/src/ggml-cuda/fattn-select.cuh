#pragma once

#include "common.cuh"

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_DEFAULT =  -1, // only returned by the per-architecture hooks: no override, keep the generic choice
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,

    // Kernels below are RDNA-only and live outside fattn.cu, see fattn-rdna.cu.
    BEST_FATTN_KERNEL_RDNA_FIRST       = 1000,
    BEST_FATTN_KERNEL_RDNA_TILE_ALLMMA = 1000,
    BEST_FATTN_KERNEL_RDNA_RTILE       = 1001,
};

static bool best_fattn_kernel_is_rdna(const best_fattn_kernel kernel) {
    return kernel >= BEST_FATTN_KERNEL_RDNA_FIRST;
}

// Everything the generic selection in fattn.cu has already derived about a FLASH_ATTN_EXT node.
// The node itself passed all generic support checks by the time a hook sees it.
struct fattn_props {
    const ggml_tensor * dst;
    int  cc;
    int  gqa_ratio;
    int  gqa_ratio_eff;         // gqa_ratio reduced to a power of 2, capped by the largest ncols2 for this head size
    bool gqa_opt_applies;
    bool can_use_vector_kernel;
    bool has_alibi;
};

// Architecture-specific overrides. Return BEST_FATTN_KERNEL_DEFAULT to keep the generic choice.
best_fattn_kernel ggml_cuda_get_best_fattn_kernel_rdna(const fattn_props & props);

// Only called for kernels with best_fattn_kernel_is_rdna(kernel):
void ggml_cuda_flash_attn_ext_rdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst, best_fattn_kernel kernel);
void ggml_cuda_fattn_need_f16_rdna(best_fattn_kernel kernel, const ggml_tensor * K, const ggml_tensor * V, bool & need_f16_K, bool & need_f16_V);
