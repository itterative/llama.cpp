// fattn-rtile: RDNA-tuned tile flash-attention decode kernel.
//
// Method (the fattn-tile dataflow, written fresh for gfx1100): block-
// cooperative K/V staging through LDS in nbatch_fa-row tiles, warps own the
// gqa-packed Q columns, lane-owns-row KQ dots (every dot lane-local, no
// cross-lane traffic in the hot loop), online softmax, V@P accumulate in
// half2 registers, launch_fattn's KV split + flash_attn_combine_results
// finish. Lane-owns-row is what keeps the LDS pipe free of ds_permute
// reduction traffic, so Q lives in LDS and is re-broadcast per chunk.
// q8_0 K is staged COMPACT (raw quants + scales, split arrays overlaid on
// the V tile storage) and dotted with dp4a against Q requantized to int8
// once per kernel (per-32-block scales, dec's numerics): half the dot
// instructions of the f16 path, 544 B/row of K LDS instead of 1024.
//
// Scope: D in {128, 256, 512}, f16/q8_0/q4_0 K/V, nb=1, even gqa >= 2,
// kv % 32 == 0, no softcap, no alibi. Design + measured basis:
// .pi/local/fattn-rtile-design-2026-08-07.md.
#include "common.cuh"
#include "fattn-common.cuh"

// Host dispatch entry (fattn-rtile.cu).
void ggml_cuda_flash_attn_ext_rtile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Base-2 softmax (dec's GGML_FATTN_DEC_EXP2 recipe): exp2f is one VALU op
// where expf is a multiply plus an exp2. INVARIANT: every additive
// contribution to a logit is pre-scaled by log2(e) -- the Q scale, the mask,
// the attention sink and FATTN_KQ_MAX_OFFSET. dst_meta's KQ_max is converted
// back to base e on the way out because flash_attn_combine_results does
// expf(meta.x - kqmax). A/B with -DGGML_FATTN_RTILE_EXP2=0.
#ifndef GGML_FATTN_RTILE_EXP2
#define GGML_FATTN_RTILE_EXP2 1
#endif

static __device__ __forceinline__ float rtile_exp(const float x) {
    return GGML_FATTN_RTILE_EXP2 ? exp2f(x) : expf(x);
}

// Config is derived, not tabled: nwarps = min(4, ncols2) so every warp owns
// cpw = ncols2/nwarps columns with np == 1 (no cross-warp combine epilogue).
// nbatch_fa = 32 = warp_size: exactly one row per lane in the KQ dot.
static constexpr __host__ __device__ int rtile_nwarps_v(const int ncols2) {
    return ncols2 < 4 ? ncols2 : 4;
}

// Stage nbatch_fa rows x D halves from global into the shared KV tile.
// f16 rows are copied verbatim; q8_0 rows are dequantized at the staging
// store, so the sweep below sees f16-width data either way -- same tile,
// same accesses, dequant VALU free (kvtype law 1). Global loads issue as
// register bursts before the LDS stores, and the +4 half2 row pad keeps
// lane-owns-row b128 reads at the 4-phase floor (528 B row stride).
// sparse staging: the tile's rows come from an index list instead of a contiguous range.
// A -1 entry (padded past the selection) or a row past the end of a short tile reads row 0,
// which is safe to touch and gets -INF from the mask, so it cannot change the result.
static __device__ __forceinline__ int rtile_row(const int * const __restrict__ rows, const int nvalid, const int row) {
    if (rows == nullptr) {
        return row;
    }
    const int r = row < nvalid ? rows[row] : -1;
    return r < 0 ? 0 : r;
}

template<int D, int nbatch_fa, int nthreads, ggml_type type_KV>
static __device__ __forceinline__ void rtile_stage_tile(
        const char * const __restrict__ src, half2 * const __restrict__ dst, const int64_t stride_bytes,
        const int * const __restrict__ rows, const int nvalid) {
    constexpr int PAD_H2 = 4;
    const int tid = threadIdx.y*warpSize + threadIdx.x;

    if constexpr (type_KV == GGML_TYPE_F16) {
        constexpr int CPR    = (D/2)/4;               // b128 chunks per row
        constexpr int NRPT   = nbatch_fa*CPR/nthreads;
        constexpr int BURST  = NRPT < 8 ? NRPT : 8;
        static_assert(nbatch_fa*CPR % nthreads == 0, "bad staging split");
        static_assert(NRPT % BURST == 0, "bad burst split");

        const half2 * src_h2 = (const half2 *) src;
        const int     stride_h2 = stride_bytes / int(sizeof(half2));
#pragma unroll
        for (int g = 0; g < NRPT/BURST; ++g) {
            __align__(16) half2 buf[BURST][4];
#pragma unroll
            for (int it = 0; it < BURST; ++it) {
                const int u = tid + (g*BURST + it)*nthreads;
                const int row = u / CPR, chunk = u % CPR;
                ggml_cuda_memcpy_1<16>(buf[it], src_h2 + (int64_t) rtile_row(rows, nvalid, row)*stride_h2 + chunk*4);
            }
#pragma unroll
            for (int it = 0; it < BURST; ++it) {
                const int u = tid + (g*BURST + it)*nthreads;
                const int row = u / CPR, chunk = u % CPR;
                ggml_cuda_memcpy_1<16>(dst + row*(D/2 + PAD_H2) + chunk*4, buf[it]);
            }
        }
    } else if constexpr (type_KV == GGML_TYPE_Q8_0) {
        static_assert(D % QK8_0 == 0, "bad q8_0 row split");
        // One work item per (row, 8-element fragment): consecutive lanes cover
        // consecutive bytes of one row, so the layout-forced 2 B transfers sit
        // ~8 B apart per lane instead of a whole 34 B block per lane. Raw
        // loads issue as one register burst before the dequant+store phase.
        constexpr int NFR = D/8;                 // fragments per row
        constexpr int NW  = nbatch_fa*NFR/nthreads;
        static_assert(nbatch_fa*NFR % nthreads == 0, "bad q8_0 staging split");

        int2 qs_raw[NW];
        half d_raw[NW];
#pragma unroll
        for (int it = 0; it < NW; ++it) {
            const int u = tid + it*nthreads;
            const int row = u / NFR, frag = u % NFR;
            const char * blk = src + (int64_t) rtile_row(rows, nvalid, row)*stride_bytes + (frag/(QK8_0/8))*sizeof(block_q8_0);
            ggml_cuda_memcpy_1<2>(&d_raw[it], blk);
            ggml_cuda_memcpy_1<8, 2>(&qs_raw[it], blk + sizeof(half) + (frag%(QK8_0/8))*8);
        }
#pragma unroll
        for (int it = 0; it < NW; ++it) {
            const int u = tid + it*nthreads;
            const int row = u / NFR, frag = u % NFR;
            const int8_t * q8 = (const int8_t *) &qs_raw[it];
            const half2    d2 = __half2half2(d_raw[it]);
            __align__(16) half2 tmp[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                tmp[l] = d2 * make_half2(q8[2*l + 0], q8[2*l + 1]);
            }
            ggml_cuda_memcpy_1<16>(dst + row*(D/2 + PAD_H2) + frag*4, tmp);
        }
    } else {
        static_assert(type_KV == GGML_TYPE_Q4_0, "fattn-rtile: K/V must be f16, q8_0 or q4_0");
        static_assert(D % QK4_0 == 0, "bad q4_0 row split");
        // Same fragment items as q8_0. A fragment is 8 consecutive elements =
        // one nibble of each of 8 consecutive bytes: byte j of a block holds
        // element j in its low nibble and j+16 in its high (dec's recipe), so
        // the load is the same b64, followed by the nibble unpack to int8.
        constexpr int NFR = D/8;
        constexpr int NW  = nbatch_fa*NFR/nthreads;
        static_assert(nbatch_fa*NFR % nthreads == 0, "bad q4_0 staging split");

        int2 qs_raw[NW];
        half d_raw[NW];
#pragma unroll
        for (int it = 0; it < NW; ++it) {
            const int u = tid + it*nthreads;
            const int row = u / NFR, frag = u % NFR;
            const char * blk = src + (int64_t) rtile_row(rows, nvalid, row)*stride_bytes + (frag/(QK4_0/8))*sizeof(block_q4_0);
            ggml_cuda_memcpy_1<2>(&d_raw[it], blk);
            ggml_cuda_memcpy_1<8, 2>(&qs_raw[it], blk + sizeof(half) + (frag & 1)*8);
        }
#pragma unroll
        for (int it = 0; it < NW; ++it) {
            const int u = tid + it*nthreads;
            const int row = u / NFR, frag = u % NFR;
            const int shift = (frag & 2)*2; // 0 for elements 0-15, 4 for 16-31
            int2 q8i;
            q8i.x = __vsubss4((qs_raw[it].x >> shift) & 0x0F0F0F0F, 0x08080808);
            q8i.y = __vsubss4((qs_raw[it].y >> shift) & 0x0F0F0F0F, 0x08080808);
            const int8_t * q8 = (const int8_t *) &q8i;
            const half2    d2 = __half2half2(d_raw[it]);
            __align__(16) half2 tmp[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                tmp[l] = d2 * make_half2(q8[2*l + 0], q8[2*l + 1]);
            }
            ggml_cuda_memcpy_1<16>(dst + row*(D/2 + PAD_H2) + frag*4, tmp);
        }
    }
}

// q8_0/q4_0 K staging, compact: quants and scales split into separate LDS
// arrays (overlaid on the V tile storage, see the kernel body), NOT
// dequantized -- the dp4a dot consumes raw int8. Same (row, 8-element
// fragment) work items as the V loader above: dense row reads, one register
// burst before the stores. q4_0 fragments are nibble-unpacked to int8 at
// the store (dec's recipe), so the dot is type-blind. K_qs rows are padded
// to 272 B so the dot's b128 reads stay conflict-free (8 lanes x 16 B per
// phase at a 4-bank stride).
template<int D, int nbatch_fa, int nthreads, ggml_type type_KV>
static __device__ __forceinline__ void rtile_stage_k_quant(
        const char * const __restrict__ src, int * const __restrict__ K_qs, half * const __restrict__ K_ds,
        const int64_t stride_bytes, const int * const __restrict__ rows, const int nvalid) {
    static_assert(type_KV == GGML_TYPE_Q8_0 || type_KV == GGML_TYPE_Q4_0, "bad K type");
    static_assert(D % QK8_0 == 0, "bad quant row split"); // QK4_0 == QK8_0 == 32
    constexpr int NFR = D/8;                 // fragments per row
    constexpr int NW  = nbatch_fa*NFR/nthreads;
    static_assert(nbatch_fa*NFR % nthreads == 0, "bad quant staging split");
    constexpr int BLK = type_KV == GGML_TYPE_Q8_0 ? int(sizeof(block_q8_0)) : int(sizeof(block_q4_0));

    const int tid = threadIdx.y*warpSize + threadIdx.x;

    int2 qs_raw[NW];
    half d_raw[NW];
#pragma unroll
    for (int it = 0; it < NW; ++it) {
        const int u = tid + it*nthreads;
        const int row = u / NFR, frag = u % NFR;
        const char * blk = src + (int64_t) rtile_row(rows, nvalid, row)*stride_bytes + (frag/4)*BLK;
        ggml_cuda_memcpy_1<2>(&d_raw[it], blk);
        if constexpr (type_KV == GGML_TYPE_Q8_0) {
            ggml_cuda_memcpy_1<8, 2>(&qs_raw[it], blk + sizeof(half) + (frag%4)*8);
        } else {
            ggml_cuda_memcpy_1<8, 2>(&qs_raw[it], blk + sizeof(half) + (frag & 1)*8);
        }
    }
#pragma unroll
    for (int it = 0; it < NW; ++it) {
        const int u = tid + it*nthreads;
        const int row = u / NFR, frag = u % NFR;
        int2 q8i = qs_raw[it];
        if constexpr (type_KV == GGML_TYPE_Q4_0) {
            const int shift = (frag & 2)*2; // 0 for elements 0-15, 4 for 16-31
            q8i.x = __vsubss4((q8i.x >> shift) & 0x0F0F0F0F, 0x08080808);
            q8i.y = __vsubss4((q8i.y >> shift) & 0x0F0F0F0F, 0x08080808);
        }
        *(int2 *) &K_qs[row*(D/4 + 4) + frag*2] = q8i;
        if ((frag & 3) == 0) { // one lane per block stores the scale
            K_ds[row*(D/QK8_0) + frag/4] = d_raw[it];
        }
    }
}

template<int D, int ncols2, ggml_type type_KV, bool use_sparse>
__launch_bounds__(rtile_nwarps_v(ncols2)*32)
static __global__ void flash_attn_rtile(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int nwarps    = rtile_nwarps_v(ncols2);
    constexpr int cpw       = ncols2/nwarps;   // Q columns per warp
    constexpr int nbatch_fa = 32;              // KV rows per tile
    constexpr int PAD_H2    = 4;
    constexpr int DL2       = D/2/32;          // half2 per lane of a D-row (4 at D=256)
    // D=512 sub-chunks every tile into 256-element halves; each half-pass
    // below is the D=256 machinery. NSUB == 1 at D=256 (identical codegen).
    static_assert(D == 128 || D == 256 || D == 512, "fattn-rtile: D must be 128, 256 or 512");
    constexpr int Dh   = D == 512 ? 256 : D;   // elements per sub-chunk
    constexpr int NSUB = D/Dh;                 // sub-chunks per tile
    constexpr int DL2h = Dh/2/32;              // half2 per lane of a sub-chunk

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    // one pointer argument, two meanings: lengths when dense, index rows when sparse
    const int  * GGML_CUDA_RESTRICT KV_max   = use_sparse ? nullptr : KV_max_ptr;
    const int  * GGML_CUDA_RESTRICT idx_all  = use_sparse ? KV_max_ptr : nullptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Index contract with launch_fattn (identical to fattn-tile): blockIdx.x is the Q tile,
    // blockIdx.y the KV split and blockIdx.z the (head group, sequence) pair.
    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0 = blockIdx.z*ncols2 - sequence*ne02;
    const int gqa_ratio = ne02 / ne12;
    const int q_idx  = blockIdx.x;
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02*head0 + nb01*q_idx);
    const char  * K_d  = K + nb13*sequence + nb12*(head0 / gqa_ratio);
    const char  * V_d  = V + nb23*sequence + nb22*(head0 / gqa_ratio);
    const half  * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*q_idx) : nullptr;

    const int64_t stride_K = nb11; // byte strides; rows are f16 or q8_0
    const int64_t stride_V = nb21;

    // launch_fattn passes n_kv_max in ne11 when the op is sparse, and one int32 row of
    // gathered indices per (sequence, query) tile in idx_all
    const int * idx_row = use_sparse ? idx_all + (int64_t((sequence % ne33)*ne31 + q_idx) * ne11) : nullptr;

    constexpr float L2E = GGML_FATTN_RTILE_EXP2 ? 1.44269504088896340736f : 1.0f;
    constexpr float KQ_MAX_OFF = FATTN_KQ_MAX_OFFSET*L2E;
    const     float scale_e2 = scale*L2E;

    // Q_tmp: staged Q columns (scale applied), read as per-chunk broadcasts;
    // f16 at NSUB > 1 stages one sub-chunk per tile-half in the K loop; at
    // quant it holds Q_q8 (int8 quants, full height) + Q_d (per-block
    // scales, softmax scale folded in). KV_tmp: K tile then V tile, shared,
    // padded rows; one sub-chunk wide at D=512, and at quant it holds the
    // compact K_qs/K_ds while K is live, then the dequantized V tile.
    // KQ: softmax'd P values, written lane-owns-row, read as per-row broadcasts.
    constexpr int Q_H2  = type_KV == GGML_TYPE_F16 ? (NSUB == 1 ? ncols2*D/2 : ncols2*Dh/2)
                        : (D == 256 ? ncols2*D/2 : ncols2*(D/4 + D/QK8_0));
    constexpr int KV_H2 = nbatch_fa*(Dh/2 + PAD_H2);
    __shared__ __align__(16) half2 Q_tmp[Q_H2];
    __shared__ __align__(16) half2 KV_tmp[KV_H2];
    __shared__ __align__(16) half  KQ[ncols2*nbatch_fa];

    int   * Q_q8 = (int *) Q_tmp;                           // ncols2 x D/4 int32
    float * Q_d  = (float *) (Q_q8 + ncols2*(D/4));         // ncols2 x D/QK8_0
    int   * K_qs = (int *) KV_tmp;                          // nbatch_fa x (Dh/4 + 4) int32
    half  * K_ds = (half *) (K_qs + nbatch_fa*(Dh/4 + 4));  // nbatch_fa x Dh/QK8_0

    __align__(16) half2 VKQ[cpw][DL2] = {{0.0f, 0.0f}};
    float KQ_max[cpw], KQ_sum[cpw] = {0.0f};
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max[jc0] = -FLT_MAX/2.0f;
    }

    ggml_cuda_pdl_sync();

    // Stage Q: lane covers d = threadIdx.x*2*DL2 .. +2*DL2-1 of each of its
    // warp's columns (8 elements at D=256, 16 at D=512). f16 at NSUB > 1
    // skips this: Q halves are staged per tile-half in the K loop below.
    if constexpr (type_KV == GGML_TYPE_F16) {
        if constexpr (NSUB == 1) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
                const int jc = threadIdx.y*cpw + jc0;
                const float * q = Q_f + jc*(nb02/sizeof(float));

                __align__(16) float qf[2*DL2];
#pragma unroll
                for (int l = 0; l < DL2/2; ++l) {
                    ggml_cuda_memcpy_1<16>(qf + 4*l, q + threadIdx.x*2*DL2 + 4*l);
                }

                __align__(16) half2 qh[DL2];
#pragma unroll
                for (int l = 0; l < DL2; ++l) {
                    qh[l] = make_half2(scale_e2*qf[2*l + 0], scale_e2*qf[2*l + 1]);
                }
                constexpr int ST = DL2 < 4 ? DL2 : 4; // store width in half2 (8 B at D=128)
#pragma unroll
                for (int l = 0; l < DL2/ST; ++l) {
                    ggml_cuda_memcpy_1<4*ST>(&Q_tmp[jc*(D/2) + threadIdx.x*DL2 + ST*l], qh + ST*l);
                }
            }
        }
    } else {
        // Requantize Q to int8 per 32-element block (dec's dp4a numerics;
        // the softmax scale folds into Q_d). LPB lanes share one block.
        constexpr int LPB = 16/DL2; // 4 at D=256, 2 at D=512
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = threadIdx.y*cpw + jc0;
            const float * q = Q_f + jc*(nb02/sizeof(float));

            __align__(16) float qf[2*DL2];
#pragma unroll
            for (int l = 0; l < DL2/2; ++l) {
                ggml_cuda_memcpy_1<16>(qf + 4*l, q + threadIdx.x*2*DL2 + 4*l);
            }

            float amax = 0.0f;
#pragma unroll
            for (int l = 0; l < 2*DL2; ++l) {
                amax = fmaxf(amax, fabsf(qf[l]));
            }
#pragma unroll
            for (int off = LPB/2; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, warpSize));
            }
            const float dq = amax/127.0f;
            const float id = dq > 0.0f ? 1.0f/dq : 0.0f;

            __align__(16) int8_t q8b[2*DL2];
#pragma unroll
            for (int l = 0; l < 2*DL2; ++l) {
                q8b[l] = (int8_t) __float2int_rn(qf[l]*id);
            }
            if constexpr (DL2 >= 4) {
#pragma unroll
                for (int l = 0; l < DL2/4; ++l) {
                    *(int2 *) &Q_q8[jc*(D/4) + threadIdx.x*(DL2/2) + l*2] = *(const int2 *) (q8b + 8*l);
                }
            } else { // DL2 == 2: 4 quants per lane, one 4 B store
                Q_q8[jc*(D/4) + threadIdx.x] = *(const int *) q8b;
            }
            if ((threadIdx.x & (LPB - 1)) == 0) {
                Q_d[jc*(D/QK8_0) + threadIdx.x/LPB] = scale_e2*dq;
            }
        }
    }

    __syncthreads();

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    // Row bytes per 256-element sub-chunk (K and V share the type).
    constexpr int64_t half_bytes = type_KV == GGML_TYPE_F16  ? Dh*int64_t(sizeof(half)) :
                                   type_KV == GGML_TYPE_Q8_0 ? (Dh/QK8_0)*int64_t(sizeof(block_q8_0))
                                                             : (Dh/QK4_0)*int64_t(sizeof(block_q4_0));
    for (int k0 = blockIdx.y*nbatch_fa; k0 < k_VKQ_max; k0 += gridDim.y*nbatch_fa) {
        // in the sparse case k0 counts selected entries, not KV rows, and the last tile is short
        const int  nvalid = use_sparse ? min(nbatch_fa, k_VKQ_max - k0) : nbatch_fa;
        const int * rows  = use_sparse ? idx_row + k0 : nullptr;
        // K in NSUB sub-chunks of Dh elements; KQ_acc accumulates across
        // them. NSUB == 1 at D=256: the sequence below is the original one.
        float KQ_acc[cpw] = {0.0f};
#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
            if constexpr (type_KV == GGML_TYPE_F16 && NSUB > 1) {
                // Stage Q half h: lane covers 8 elements of each column's
                // half. Covered by the K-stage barrier below.
#pragma unroll
                for (int jc0 = 0; jc0 < cpw; ++jc0) {
                    const int jc = threadIdx.y*cpw + jc0;
                    const float * q = Q_f + jc*(nb02/sizeof(float)) + h*Dh;
                    __align__(16) float qf[2*DL2h];
#pragma unroll
                    for (int l = 0; l < DL2h/2; ++l) {
                        ggml_cuda_memcpy_1<16>(qf + 4*l, q + threadIdx.x*2*DL2h + 4*l);
                    }
                    __align__(16) half2 qh[DL2h];
#pragma unroll
                    for (int l = 0; l < DL2h; ++l) {
                        qh[l] = make_half2(scale_e2*qf[2*l + 0], scale_e2*qf[2*l + 1]);
                    }
                    ggml_cuda_memcpy_1<16>(&Q_tmp[jc*(Dh/2) + threadIdx.x*DL2h], qh);
                }
            }
            if constexpr (type_KV == GGML_TYPE_F16) {
                rtile_stage_tile<Dh, nbatch_fa, nwarps*32, type_KV>(K_d + h*half_bytes + (use_sparse ? 0 : (int64_t) k0*stride_K), KV_tmp, stride_K, rows, nvalid);
            } else {
                rtile_stage_k_quant<Dh, nbatch_fa, nwarps*32, type_KV>(K_d + h*half_bytes + (use_sparse ? 0 : (int64_t) k0*stride_K), K_qs, K_ds, stride_K, rows, nvalid);
            }
            __syncthreads();

            // KQ dot: lane owns row threadIdx.x. f16: serial over the row in
            // b128 chunks (grouping this loop helped neither at f16 (v7) nor
            // at q8_0 (v2dot)). quant: dp4a on raw quants, one 32-element
            // block at a time, Q int8 broadcast from LDS, block scales
            // combined per block.
            if constexpr (type_KV == GGML_TYPE_F16) {
#pragma unroll
                for (int kk = 0; kk < Dh/2; kk += 4) {
                    __align__(16) half2 K_k[4];
                    __align__(16) half2 Q_k[cpw][4];
                    ggml_cuda_memcpy_1<16>(K_k, &KV_tmp[threadIdx.x*(Dh/2 + PAD_H2) + kk]);
#pragma unroll
                    for (int jc0 = 0; jc0 < cpw; ++jc0) {
                        ggml_cuda_memcpy_1<16>(Q_k[jc0], &Q_tmp[(threadIdx.y*cpw + jc0)*(Q_H2/ncols2) + kk]);
                    }
#pragma unroll
                    for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            ggml_cuda_mad(KQ_acc[jc0], K_k[l], Q_k[jc0][l]);
                        }
                    }
                }
            } else {
                // Runtime block loop: fully unrolled, the hoisted k_w/q_w
                // arrays blow the 256-vgpr budget (v6: 54-reg spill,
                // scratch=220, 478 us at 131k). The runtime body keeps the
                // load burst and constant register indices; d_all would be a
                // runtime index here, so the scale comes from LDS per block.
                const int * K_row = K_qs + threadIdx.x*(Dh/4 + 4);
#pragma unroll 1
                for (int b = 0; b < Dh/QK8_0; ++b) {
                    __align__(16) int k_w[8];
                    ggml_cuda_memcpy_1<16>(k_w + 0, K_row + b*8 + 0);
                    ggml_cuda_memcpy_1<16>(k_w + 4, K_row + b*8 + 4);
                    const float dk = __half2float(K_ds[threadIdx.x*(Dh/QK8_0) + b]);
#pragma unroll
                    for (int jc0 = 0; jc0 < cpw; ++jc0) {
                        const int jc = threadIdx.y*cpw + jc0;
                        __align__(16) int q_w[8];
                        ggml_cuda_memcpy_1<16>(q_w + 0, Q_q8 + jc*(D/4) + h*(Dh/4) + b*8 + 0);
                        ggml_cuda_memcpy_1<16>(q_w + 4, Q_q8 + jc*(D/4) + h*(Dh/4) + b*8 + 4);
                        int sumi = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sumi = ggml_cuda_dp4a(k_w[i], q_w[i], sumi);
                        }
                        KQ_acc[jc0] += dk*Q_d[jc*(D/QK8_0) + h*(Dh/QK8_0) + b]*(float) sumi;
                    }
                }
            }
            if (h + 1 < NSUB) {
                __syncthreads(); // this half's K reads done before the next stage
            }
        }

        // Mask, online-softmax max:
        float KQ_max_new[cpw];
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            if constexpr (use_sparse) {
                // the selection is per token, so every Q column in the tile shares one index row
                const int idx = threadIdx.x < nvalid ? rows[threadIdx.x] : -1;
                KQ_acc[jc0] += threadIdx.x < nvalid && idx >= 0 ? __half2float(maskh[idx])*L2E : -INFINITY;
            } else {
                KQ_acc[jc0] += maskh ? __half2float(maskh[k0 + threadIdx.x])*L2E : 0.0f;
            }
            KQ_max_new[jc0] = fmaxf(KQ_max[jc0], KQ_acc[jc0] + KQ_MAX_OFF);
            KQ_max_new[jc0] = warp_reduce_max<32>(KQ_max_new[jc0]);
        }

        // exp, sum, P store, VKQ rescale:
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const float KQ_max_scale = rtile_exp(KQ_max[jc0] - KQ_max_new[jc0]);
            KQ_max[jc0] = KQ_max_new[jc0];

            const float val = rtile_exp(KQ_acc[jc0] - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;
            KQ[(threadIdx.y*cpw + jc0)*nbatch_fa + threadIdx.x] = __float2half(val);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int l = 0; l < DL2; ++l) {
                VKQ[jc0][l] *= KQ_max_scale_h2;
            }
        }
        __syncthreads(); // K reads + P writes done before V stage overwrites KV_tmp

#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
            rtile_stage_tile<Dh, nbatch_fa, nwarps*32, type_KV>(V_d + h*half_bytes + (use_sparse ? 0 : (int64_t) k0*stride_V), KV_tmp, stride_V, rows, nvalid);
            __syncthreads();

            // VKQ accumulate: lane owns d-slice threadIdx.x*2*DL2h .. +2*DL2h-1
            // of the sub-chunk. Rows run in runtime groups of VG (do NOT
            // unroll the group loop: fully unrolled, the scheduler
            // interleaves each load with the previous row's FMAs -- depth
            // zero, 40/40 full drains. dec_lds's dot gets its burst from
            // exactly this runtime-loop body shape). P for a group is 8 rows
            // == one b128 per column, loaded with compile-time-constant
            // register indices only (a dynamic index would push the block to
            // scratch).
            constexpr int VG = 8;
            static_assert(nbatch_fa % VG == 0, "bad V burst group");
#pragma unroll 1
            for (int k0v = 0; k0v < nbatch_fa; k0v += VG) {
                half2 P_g[cpw][VG/2];
#pragma unroll
                for (int jc0 = 0; jc0 < cpw; ++jc0) {
                    ggml_cuda_memcpy_1<16>(P_g[jc0], &KQ[(threadIdx.y*cpw + jc0)*nbatch_fa + k0v]);
                }
                __align__(16) half2 V_k[VG][DL2h];
#pragma unroll
                for (int g = 0; g < VG; ++g) {
                    ggml_cuda_memcpy_1<4*DL2h>(V_k[g], &KV_tmp[(k0v + g)*(Dh/2 + PAD_H2) + threadIdx.x*DL2h]);
                }
#pragma unroll
                for (int g = 0; g < VG; ++g) {
#pragma unroll
                    for (int jc0 = 0; jc0 < cpw; ++jc0) {
                        const half  pr = (g & 1) ? P_g[jc0][g/2].y : P_g[jc0][g/2].x;
                        const half2 p  = __half2half2(pr);
#pragma unroll
                        for (int l = 0; l < DL2h; ++l) {
                            VKQ[jc0][h*DL2h + l] += V_k[g][l]*p;
                        }
                    }
                }
            }
            __syncthreads(); // V reads done before the next stage overwrites
        }
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum<32>(KQ_sum[jc0]);
    }

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const float sink = ((const float *) sinks)[head0 + threadIdx.y*cpw + jc0]*L2E;

            const float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = rtile_exp(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + rtile_exp(sink - KQ_max[jc0]);

            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int l = 0; l < DL2; ++l) {
                VKQ[jc0][l] *= KQ_max_scale_h2;
            }
        }
    }

    // Write back results (ncols1 == 1: j == 0, c == jc, no column guard):
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = threadIdx.y*cpw + jc0;
        const float scale_o = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;
        const int j_dst = ((sequence*int(ne01.z) + q_idx)*ne02 + head0 + jc)*gridDim.y + blockIdx.y;

        __align__(16) float2 tmp[DL2];
#pragma unroll
        for (int l = 0; l < DL2; ++l) {
            tmp[l] = __half22float2(VKQ[jc0][l]);
            tmp[l].x *= scale_o;
            tmp[l].y *= scale_o;
        }
#pragma unroll
        for (int h = 0; h < NSUB; ++h) {
#pragma unroll
            for (int l = 0; l < DL2h; l += 2) {
                // VKQ[h*DL2h + l] holds elements h*Dh + tid*2*DL2h + 2l (the
                // sweep's per-half lane slice, NOT tid*2*DL2 contiguous).
                ggml_cuda_memcpy_1<16>(&dst[j_dst*D + h*Dh + threadIdx.x*2*DL2h + l*2], tmp + h*DL2h + l);
            }
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            // combine_results does expf(meta.x - kqmax), so hand it base-e units.
            dst_meta[j_dst] = make_float2(KQ_max[jc0]*(GGML_FATTN_RTILE_EXP2 ? 0.693147180559945309417f : 1.0f), KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// host dispatch, defined in fattn-rtile.cu, called from fattn-rdna.cu:
void ggml_cuda_flash_attn_ext_rtile(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
