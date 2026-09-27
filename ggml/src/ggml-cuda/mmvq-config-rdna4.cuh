// RDNA4 (gfx12) block shape for mul_mat_vec_q. Same role as the mmq-config-*.cuh tables.
// Both knobs are compile time: mul_mat_vec_q takes nwarps and rows_per_cuda_block as constexpr,
// so a sweep costs a rebuild per point rather than an env var.

#ifndef MMVQ_RDNA4_NWARPS
#define MMVQ_RDNA4_NWARPS 8
#endif

// rows_per_block = MMVQ_RDNA4_ROWS_MULT * nwarps when small_k.
#ifndef MMVQ_RDNA4_ROWS_MULT
#define MMVQ_RDNA4_ROWS_MULT 1
#endif

// Only blocks_per_row_x * (qi/vdr) threads enter the K loop, so a narrow K leaves a wide block
// mostly idle and still paying the full cross-warp reduction. Cut nwarps to match instead.
#ifndef MMVQ_RDNA4_NARROW_K_MAX
#define MMVQ_RDNA4_NARROW_K_MAX 5    // blocks_per_row_x <= this -> 1 warp
#endif
#ifndef MMVQ_RDNA4_MID_K_MAX
#define MMVQ_RDNA4_MID_K_MAX 20      // blocks_per_row_x <= this -> 2 warps
#endif

// nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
// Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
// pressure and lookup table contention at higher thread counts.
static constexpr __host__ __device__ bool mmvq_rdna4_wide_block_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// q6_K is the one type that loses: 46.19 -> 55.57 us at k=1024 on RX 9070, and its best warp count
// is not monotone in K (1 at kblk 1, 8 at kblk 4, 4 at kblk 10), so it keeps the table value.
static constexpr __host__ __device__ bool mmvq_rdna4_narrow_k_type(ggml_type type) {
    return type != GGML_TYPE_Q6_K;
}

// narrow_nwarps is chosen by the host from blocks_per_row_x, 0 means use the table above.
static constexpr __host__ __device__ int mmvq_calc_nwarps_rdna4(ggml_type type, int ncols_dst, int narrow_nwarps = 0) {
    if (narrow_nwarps > 0) {
        return narrow_nwarps;
    }
    return (ncols_dst == 1 && mmvq_rdna4_wide_block_type(type)) ? MMVQ_RDNA4_NWARPS : 1;
}

// Must stay <= warp_size: the epilogue writes row i from threadIdx.x == i.
static constexpr __host__ __device__ int mmvq_calc_rows_per_block_rdna4(int ncols_dst, bool small_k, int nwarps) {
    switch (ncols_dst) {
        case 1:
            return small_k ? MMVQ_RDNA4_ROWS_MULT*nwarps : 1;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
            return 2;
        default:
            return 1;
    }
}
