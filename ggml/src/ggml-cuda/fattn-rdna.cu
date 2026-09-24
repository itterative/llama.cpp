// RDNA-specific FlashAttention kernel selection and dispatch.
// fattn.cu asks this file for an override after it has run the generic support checks;
// returning BEST_FATTN_KERNEL_DEFAULT leaves the generic choice untouched.

#include "fattn-select.cuh"

#if defined(GGML_USE_HIP)

#include "fattn-rtile.cuh"

#include <cstdlib>

static bool ggml_cuda_fattn_rdna_can_use_rtile(const fattn_props & props) {
    const ggml_tensor * Q = props.dst->src[0];
    const ggml_tensor * K = props.dst->src[1];
    const ggml_tensor * V = props.dst->src[2];

    if (!GGML_CUDA_CC_IS_RDNA3(props.cc) && !GGML_CUDA_CC_IS_RDNA4(props.cc)) {
        return false;
    }
    if (K->ne[0] != V->ne[0] || (K->ne[0] != 128 && K->ne[0] != 256 && K->ne[0] != 512)) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_Q4_0)) {
        return false;
    }
    if (props.has_alibi) {
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) props.dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }

    // ncols2 is the largest of {16,8,4,2} dividing gqa
    // one block per Q token, so a dense op would re-read the whole cache per query and is left to
    // the kernels that pack queries; a sparse op gathers a private index row per query, where
    // separate blocks are not redundant work. 16 covers speculative verification widths.
    const int32_t n_kv_max = ggml_get_op_params_i32(props.dst, 4);
    const int64_t n_q_max = n_kv_max > 0 ? 16 : 1;

    if (Q->ne[1] > n_q_max || Q->ne[1] < 1 || props.gqa_ratio < 2 || props.gqa_ratio % 2 != 0 ||
            K->ne[1] % 32 != 0) {
        return false;
    }

    if (n_kv_max == 0) {
        return true; // dense: walk the cache
    }

    // sparse: the index rows are read out of the mask, so its layout must be what the
    //     compaction kernel writes, and the selection only pays past a few times its own width
    const ggml_tensor * mask = props.dst->src[3];
    const int64_t depth_min = 2LL*n_kv_max > 4096 ? 2LL*n_kv_max : 4096;

    return mask && mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
           K->ne[1] >= depth_min;
}

// launch_fattn's use_sparse must be derived from the op hint: passing it true for a node with
// n_kv_max == 0 trips GGML_ASSERT(n_kv_max > 0) in fattn-common.cuh.

static bool ggml_cuda_fattn_rdna_rtile_enabled() {
    static const bool enabled = getenv("GGML_FATTN_RDNA_RTILE") && atoi(getenv("GGML_FATTN_RDNA_RTILE")) != 0;
    return enabled;
}

best_fattn_kernel ggml_cuda_get_best_fattn_kernel_rdna(const fattn_props & props) {
    if (ggml_cuda_fattn_rdna_rtile_enabled() && ggml_cuda_fattn_rdna_can_use_rtile(props)) {
        return BEST_FATTN_KERNEL_RDNA_RTILE;
    }

    return BEST_FATTN_KERNEL_DEFAULT;
}

void ggml_cuda_flash_attn_ext_rdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst, best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_RDNA_TILE_ALLMMA:
            // tile_allmma lives in a separate file that is not in this tree yet
            GGML_ABORT("fattn-rdna: tile_allmma is not available");
            break;
        case BEST_FATTN_KERNEL_RDNA_RTILE:
            ggml_cuda_flash_attn_ext_rtile(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_fattn_need_f16_rdna(best_fattn_kernel kernel, const ggml_tensor * K, const ggml_tensor * V, bool & need_f16_K, bool & need_f16_V) {
    GGML_UNUSED(K); GGML_UNUSED(V);

    switch (kernel) {
        case BEST_FATTN_KERNEL_RDNA_TILE_ALLMMA:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_RDNA_RTILE: // K/V are consumed natively, incl. q8_0/q4_0
            need_f16_K = false;
            need_f16_V = false;
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

#else

best_fattn_kernel ggml_cuda_get_best_fattn_kernel_rdna(const fattn_props & props) {
    GGML_UNUSED(props);
    return BEST_FATTN_KERNEL_DEFAULT;
}

void ggml_cuda_flash_attn_ext_rdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst, best_fattn_kernel kernel) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst); GGML_UNUSED(kernel);
    GGML_ABORT("fatal error");
}

void ggml_cuda_fattn_need_f16_rdna(best_fattn_kernel kernel, const ggml_tensor * K, const ggml_tensor * V, bool & need_f16_K, bool & need_f16_V) {
    GGML_UNUSED(kernel); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(need_f16_K); GGML_UNUSED(need_f16_V);
    GGML_ABORT("fatal error");
}

#endif // defined(GGML_USE_HIP)
