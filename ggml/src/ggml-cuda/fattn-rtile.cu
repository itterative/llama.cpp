// Host dispatch entry point for fattn-rtile (fattn-rtile.cuh).
// Scope: D=128, 256 or 512, f16, q8_0 or q4_0 K/V (consumed natively --
// need_f16 false, no upfront dequantize of the cache), nb=1, even
// gqa >= 2, kv % 32 == 0, no softcap, no alibi; ncols2 is the largest of
// {16,8,4,2} dividing gqa. D=512 runs as 256-element sub-chunks.
// Design: .pi/local/fattn-rtile-design-2026-08-07.md.
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-rtile.cuh"
#include <cstdlib>

// KV-split floor at long kv (design doc sections 24-25): launch_fattn's
// wave-fill heuristic stops at ~1 wave, which is 9-37% off at kv >= 32768.
// Measured per-cell floors (within ~2% of each cell's optimum): q4_0 144;
// q8_0 144/96/48 for ncols2 16/8/<=4; f16 144 at ncols2=16, else default.
// Below 32k the heuristic stands (1 = no-op floor). Env override for A/B:
// GGML_FATTN_RTILE_PB=N forces the floor, -1 = off.
static int rtile_min_parallel_blocks(const ggml_tensor * dst, const int ncols2) {
    static const int pb_env = [] {
        const char * s = getenv("GGML_FATTN_RTILE_PB");
        return s ? atoi(s) : 0;
    }();
    if (pb_env != 0) {
        return pb_env > 0 ? pb_env : 1;
    }
    const ggml_tensor * K = dst->src[1];
    if (K->ne[1] < 32768) {
        return 1;
    }
    if (K->type == GGML_TYPE_Q4_0) {
        return 144;
    }
    if (K->type == GGML_TYPE_Q8_0) {
        return ncols2 == 16 ? 144 : (ncols2 == 8 ? 96 : 48);
    }
    return ncols2 == 16 ? 144 : 1; // f16: only the gqa-16 cell was off ceiling
}

template <int D, int ncols2, ggml_type type_KV>
static void ggml_cuda_flash_attn_ext_rtile_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps    = rtile_nwarps_v(ncols2);
    constexpr int nbatch_fa = 32;
    constexpr size_t nbytes_shared = 0; // all state is static __shared__ inside the kernel

    // the hint is only a hint: launch_fattn sizes the grid off n_kv_max when it is set
    const bool use_sparse = ggml_get_op_params_i32(dst, 4) > 0;

    fattn_kernel_t fattn_kernel = use_sparse ? flash_attn_rtile<D, ncols2, type_KV, true>
                                             : flash_attn_rtile<D, ncols2, type_KV, false>;
    launch_fattn<D, 1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, false, false, false, use_sparse, 32,
         rtile_min_parallel_blocks(dst, ncols2));
}

template <int D, int ncols2>
static void ggml_cuda_flash_attn_ext_rtile_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    if (K->type == GGML_TYPE_F16) {
        ggml_cuda_flash_attn_ext_rtile_case<D, ncols2, GGML_TYPE_F16>(ctx, dst);
    } else if (K->type == GGML_TYPE_Q8_0) {
        ggml_cuda_flash_attn_ext_rtile_case<D, ncols2, GGML_TYPE_Q8_0>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_rtile_case<D, ncols2, GGML_TYPE_Q4_0>(ctx, dst);
    }
}

template <int D>
static void ggml_cuda_flash_attn_ext_rtile_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int gqa_ratio) {
    if (gqa_ratio % 16 == 0) {
        ggml_cuda_flash_attn_ext_rtile_type<D, 16>(ctx, dst);
    } else if (gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_rtile_type<D, 8>(ctx, dst);
    } else if (gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_rtile_type<D, 4>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_rtile_type<D, 2>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext_rtile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    GGML_ASSERT(logit_softcap == 0.0f && "fattn-rtile: no logit softcap");
    GGML_ASSERT(max_bias == 0.0f       && "fattn-rtile: no alibi");
    GGML_ASSERT(K->ne[0] == V->ne[0] && (K->ne[0] == 128 || K->ne[0] == 256 || K->ne[0] == 512) && "fattn-rtile: D=128/256/512 only");
    GGML_ASSERT(Q->ne[1] <= 16 && Q->ne[1] >= 1 && "fattn-rtile: one block per Q token, up to 16");
    GGML_ASSERT((Q->ne[1] == 1 || ggml_get_op_params_i32(KQV, 4) > 0) && "fattn-rtile: multi-Q needs a gather");
    GGML_ASSERT(K->type == V->type && "fattn-rtile: K and V must have the same type");
    GGML_ASSERT(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(K->ne[1] % 32 == 0 && "fattn-rtile: kv must be a multiple of 32");
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(gqa_ratio >= 2 && gqa_ratio % 2 == 0 && "fattn-rtile: even gqa >= 2 only");

    if (K->ne[0] == 128) {
        ggml_cuda_flash_attn_ext_rtile_gqa<128>(ctx, dst, gqa_ratio);
    } else if (K->ne[0] == 256) {
        ggml_cuda_flash_attn_ext_rtile_gqa<256>(ctx, dst, gqa_ratio);
    } else {
        ggml_cuda_flash_attn_ext_rtile_gqa<512>(ctx, dst, gqa_ratio);
    }
}
