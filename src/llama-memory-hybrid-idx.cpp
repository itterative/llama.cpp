#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// env: Q4EXP_POOLED - pool the indexer block keys at write time instead of re-deriving them every step
// on by default; setting it to 0 reverts to the historic graph and frees the pool tensors
static bool llama_qsa_pool_enabled() {
    const char * val = std::getenv("Q4EXP_POOLED");

    return val == nullptr || atoi(val) != 0;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    // one pool per ratio is only addressed append-stably while the graph writes one stream at a time,
    // so a cache that keeps a stream per sequence does not pay for it
    qsa_pool_on((unified || n_seq_max == 1) && filter_idx != nullptr && llama_qsa_pool_enabled()),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells, block key pool = %d\n", __func__, kv_size, qsa_pool_on);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_", qsa_pool_on);
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);

        qsa_pool_invalidate();
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);

        if (p1 == -1 && p0 >= 0) {
            qsa_pool_truncate(seq_id, p0);
        } else {
            qsa_pool_invalidate();
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);

        qsa_pool_invalidate();
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);

        qsa_pool_invalidate();
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);

        qsa_pool_invalidate();
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);

        qsa_pool_invalidate();
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

        // restored cells carry the positions they were saved with, which need not match the pool.
        // a per-sequence restore truncates instead of rebuilding the world: speculative rollbacks land
        // here (the recurrent cache refuses partial seq_rm), and only the blocks past the restored end
        // lose their keys
        if (mem_idx && seq_id >= 0 && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            const llama_pos p_end = mem_idx->seq_pos_max(seq_id);

            qsa_pool_truncate(seq_id, p_end + 1);
        } else {
            qsa_pool_invalidate();
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    qsa_pool_invalidate();
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::qsa_pool_truncate(llama_seq_id seq_id, llama_pos p0) {
    if (qsa_runs.empty()) {
        return;
    }

    // the pool only exists for a single-stream cache, so this is the cells array the runs were keyed on
    const int32_t j1 = (int32_t) mem_idx->get_cells(seq_id).used_max_p1();

    for (auto & [ratio, run] : qsa_runs) {
        // block b covers [(b_lo + b)*r, (b_lo + b)*r + r - 1], so it survives a cut at p0 whole if its
        // last position is below it - the rest is re-derived by the next steps as it completes again
        const int64_t n_keep = p0 >= (llama_pos) ratio
            ? (p0 - (llama_pos) ratio)/(llama_pos) ratio - (int64_t) run.b_lo + 1
            : 0;

        run.wm = std::min<uint32_t>(run.wm, n_keep > 0 ? (uint32_t) n_keep : 0);
        run.j1 = std::min(run.j1, j1);
    }
}

llama_qsa_pool llama_memory_hybrid_idx::qsa_pool_get(uint32_t ratio, const llama_ubatch & ubatch, uint32_t n_stream, uint32_t n_kv, bool worst_case) const {
    llama_qsa_pool res;

    // the pool is addressed by block index, which is only append-stable in the state the fast path of
    // set_input_qsa handles: one sequence, its used cells a dense run of consecutive positions
    if (!qsa_pool_on || ratio == 0 || n_stream != 1) {
        return res;
    }

    // graph reservation walks a full context, which holds no cells, so the state test below cannot run
    // there. answer with the widest pooled graph instead: every block derived and written. without it the
    // reservation measures the historic graph and ggml-alloc re-reserves on every pooled ubatch
    if (worst_case) {
        const uint32_t n_bid = (n_kv + ratio - 1)/ratio;

        if (n_bid == 0) {
            return res;
        }

        res.mode  = llama_qsa_pool::CACHED;
        res.n_bid = n_bid;
        res.n_new = n_bid;

        return res;
    }

    const auto & cells = mem_idx->get_cells(ubatch.seq_id[0][0]);

    int n_seq = 0;

    for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq < 2; ++sq) {
        n_seq += cells.seq_pos_min(sq) >= 0;
    }

    if (n_seq > 1) {
        return res;
    }

    const int64_t j0 = cells.used_min();
    const int64_t j1 = cells.used_max_p1();
    const int64_t nu = cells.get_used();

    const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

    if (nu == 0 || j1 - j0 != nu) {
        return res;
    }

    const llama_pos p0 = cells.pos_get(j0);
    const llama_pos p1 = cells.pos_get(j1 - 1);

    if (p0 < 0 || p1 != p0 + (llama_pos) (j1 - 1 - j0)) {
        return res;
    }

    if (p1/(llama_pos) ratio >= n_blocks) {
        return res;
    }

    // a bucket is pooled only when all of its positions are inside the run
    const uint32_t b_lo = (uint32_t) ((p0 + ratio - 1)/ratio);
    const int64_t  b_hi = p1 >= (llama_pos) (ratio - 1) ? (p1 + 1 - (llama_pos) ratio)/(llama_pos) ratio : -1;

    const uint32_t n_bid = b_hi >= (int64_t) b_lo ? (uint32_t) (b_hi - b_lo + 1) : 0;

    const auto it = qsa_runs.find(ratio);

    uint32_t wm = 0;

    bool have = false;

    if (it != qsa_runs.end()) {
        // the run must be the one the rows were keyed on. set_input_qsa verified it dense and in
        // position order through recorded j1, so only the cells appended since need looking at
        have = it->second.j0 == (int32_t) j0 && it->second.p0 == p0
            && it->second.b_lo == b_lo && it->second.wm > 0 && it->second.wm <= n_bid;

        for (int64_t j = have ? std::max<int64_t>(it->second.j1, j0) : j1; have && j < j1; ++j) {
            if (cells.pos_get(j) != p0 + (llama_pos) (j - j0)) {
                have = false;
            }
        }

        if (have) {
            wm = it->second.wm;
        }
    }

    if (!have) {
        // nothing valid recorded for this numbering: the same graph as the cached case with wm = 0, so
        // it derives and writes every complete block. one shape for both, and a cold start no longer
        // changes the node set
        res.mode  = llama_qsa_pool::CACHED;
        res.wm    = 0;
        res.n_bid = n_bid;
        // a run shorter than one block pools nothing, but still writes a row: the node set has to stay
        // the same, or the reservation that measured it is dropped. every block bias is -inf while
        // n_bid == 0, so the row cannot reach the selection
        res.n_new = std::max<uint32_t>(1, n_bid);

        return res;
    }

    res.mode  = llama_qsa_pool::CACHED;
    res.wm    = wm;
    res.n_bid = n_bid;

    // re-writing the newest row when nothing completed keeps the write tensors, and with them the
    // graph, shaped the same on every decode step
    res.n_new = std::max<uint32_t>(1, n_bid - wm);

    return res;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * blk_cells,
        ggml_tensor * tail_cells,
        ggml_tensor * cell_blk,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * new_rows,
        ggml_tensor * new_cells,
        ggml_tensor * new_pos,
        const llama_qsa_pool & pool,
        const llama_ubatch * ubatch,
        uint32_t n_kv,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    // exactly one of the two index lists is present: the tail for whole-block selection, the forward
    // map for the per-cell one
    GGML_ASSERT((tail_cells != nullptr) != (cell_blk != nullptr));
    GGML_ASSERT(tail_cells == nullptr || blk_bias);

    GGML_ASSERT(ggml_backend_buffer_is_host(blk_cells->buffer));

    const int64_t n_ns     = bias->ne[2];           // streams in this ubatch
    const int64_t r        = ratio;
    const int64_t n_blocks = blk_cells->ne[0]/r;
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(!blk_pos || blk_pos->ne[0] == 4*n_blocks*n_ns);

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = cell_blk ? (int32_t *) cell_blk->data : nullptr;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_tail      = tail_cells ? (int32_t *) tail_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos ? (int32_t *) blk_pos->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    // the pool rows the graph writes: [wm, n_bid) when it reads the pool, [0, n_bid) when it rebuilds
    int32_t * dst_new_rows  = new_rows ? (int32_t *) new_rows->data : nullptr;
    int32_t * dst_new_cells = new_cells ? (int32_t *) new_cells->data : nullptr;
    int32_t * dst_new_pos   = new_pos  ? (int32_t *) new_pos->data  : nullptr;

    GGML_ASSERT(pool.mode == llama_qsa_pool::NONE || n_ns == 1);
    GGML_ASSERT(!dst_new_rows || (int64_t) pool.n_new == new_rows->ne[0]);

    // the dense single-sequence run the pool numbering was built on, for stamping the watermark
    bool      run_fast = false;
    int32_t   run_j0   = -1;
    llama_pos run_p0   = -1;
    int32_t   run_j1   = -1;
    uint32_t  run_b_lo = 0;
    uint32_t  run_wm   = 0;

    // the graph's pool copy is from its own build, so read the live watermark for the row range
    uint32_t  wm_live  = 0;

    if (pool.mode == llama_qsa_pool::CACHED) {
        const auto it_wm = qsa_runs.find(ratio);

        wm_live = it_wm != qsa_runs.end() ? it_wm->second.wm : 0;
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of;
    std::vector<int32_t>  cell_grp;
    std::vector<int32_t>  grp_head;
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    if (dst_blk_pos) {
        std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        // per-sequence key -> cell map for the tail (Eq. 19); the fast path instead maps keys to
        // slots arithmetically, so this only serves the general path where several sequences can
        // share one cells array and a bucket does not identify the query's own sequence
        std::vector<llama_seq_id> key_seqs;
        std::vector<int32_t>      key_cell;

        int64_t fast_p0 = 0;
        int64_t fast_j0 = 0;

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            blk_of.assign(n_kv, -1);
            cell_grp.assign(n_kv, -1);
            grp_head.assign(n_blocks, -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        int32_t n_bid = 0;

        // the usual state is a single sequence whose used cells are one contiguous run of
        //     positions, where the block mapping is arithmetic - verified in a single pass, this
        //     skips the group machinery and the second walk over the cells
        auto try_contiguous = [&](void) -> bool {
            // a run with one cell per position cannot repeat a slot bit, so the ranked (mrope
            //     duplicate) branch below is unreachable whenever this one applies
            if (!blk_bias || !one_seq) {
                return false;
            }

            const int64_t j0 = cells.used_min();
            const int64_t j1 = cells.used_max_p1();
            const int64_t nu = cells.get_used();

            if (nu == 0 || j1 - j0 != nu) {
                return false;
            }

            const llama_pos p0 = cells.pos_get(j0);
            const llama_pos p1 = cells.pos_get(j1 - 1);

            if (p0 < 0 || p1 != p0 + (j1 - 1 - j0) || p1/r >= n_blocks) {
                return false;
            }

            // a bucket is pooled only when all r of its positions are inside the run
            const int64_t b_lo = (p0 + r - 1)/r;
            const int64_t b_hi = p1 >= r - 1 ? (p1 - r + 1)/r : -1;

            n_bid = b_hi >= b_lo ? (int32_t) (b_hi - b_lo + 1) : 0;

            const int32_t dead = n_bid < n_blocks ? n_bid : n_blocks - 1;

            if (cur_cell_blk) {
                std::fill(cur_cell_blk, cur_cell_blk + n_kv, dead);
            }

            // one pass: verify the run is dense and write the cell mapping as it goes. positions
            //     step by one, so the bucket and slot advance instead of being divided out
            int64_t b    = p0/r;
            int32_t slot = (int32_t) (p0%r);

            for (int64_t j = j0; j < j1; ++j) {
                if (cells.pos_get(j) != p0 + (j - j0)) {
                    return false;
                }

                if (b >= b_lo && b <= b_hi) {  // otherwise the cell is in the incomplete head or
                    cur_blk_cells[(b - b_lo)*r + slot] = (int32_t) j;

                    if (cur_cell_blk) {
                        cur_cell_blk[j] = (int32_t) (b - b_lo);   // tail bucket and uses the spare one
                    }
                }

                if (++slot == r) {
                    slot = 0;
                    b++;
                }
            }

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t idx = (int32_t) ((b_lo + b)*r);

                bid_idx .push_back(idx);
                bid_cell.push_back((int32_t) (j0 + (idx - p0)));

                for (int64_t sec = 0; sec < 4 && dst_blk_pos; ++sec) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = idx;
                }
            }

            if (dst_new_rows) {
                // rows from the live watermark up to the newest complete block. a reused graph only
                // guarantees the row count, so widen a short list by repeating the newest block - its
                // key does not change, and the duplicate lands on the row that already holds it
                const int32_t b0 = pool.mode == llama_qsa_pool::CACHED ? (int32_t) wm_live : 0;
                const int32_t nw = (int32_t) pool.n_new;

                GGML_ASSERT(nw > 0 && b0 <= n_bid && (uint32_t) (n_bid - b0) <= pool.n_new);

                for (int32_t j = 0; j < nw; ++j) {
                    const int32_t b = std::min(b0 + j, std::max(n_bid - 1, 0));

                    // no complete block yet: derive row 0 from the head of the run instead
                    const int32_t idx = n_bid > 0 ? (int32_t) ((b_lo + b)*r) : (int32_t) p0;
                    const int32_t c0  = (int32_t) (j0 + (idx - p0));

                    dst_new_rows[j] = b;

                    if (dst_new_cells) {
                        // block-major like blk_cells: the graph reads [r, n_new] with r fastest. a run
                        // shorter than r repeats its last cell
                        for (int64_t m = 0; m < r; ++m) {
                            dst_new_cells[j*r + m] = std::min(c0 + (int32_t) m, (int32_t) (j1 - 1));
                        }

                        for (int64_t sec = 0; sec < 4; ++sec) {
                            dst_new_pos[sec*nw + j] = idx;
                        }
                    }
                }

                run_wm = std::min<uint32_t>(n_bid, b0 + nw);
            }

            if (pool.mode != llama_qsa_pool::NONE) {
                run_fast   = true;
                run_j0     = (int32_t) j0;
                run_p0     = p0;
                run_j1     = (int32_t) j1;
                run_b_lo   = b_lo;
            }

            fast_p0 = p0;
            fast_j0 = j0;

            return true;
        };

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const llama_seq_id seq_id = ubatch->seq_id[s*n_tps + ii][0];

                if (std::find(key_seqs.begin(), key_seqs.end(), seq_id) == key_seqs.end()) {
                    key_seqs.push_back(seq_id);
                }
            }

            const bool fast = try_contiguous();

            if (!fast) {
            group_cells();

            // mrope repeats one position across an image, so rank cells instead of using the position
            if (dup && ubatch->is_pos_2d() && one_seq) {
                order.clear();
                order.reserve(n_kv);

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (!cells.is_empty(j)) {
                        order.push_back((int32_t) j);
                    }
                }

                // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
                std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                    const llama_pos pa = cells.pos_get(a);
                    const llama_pos pb = cells.pos_get(b);

                    if (pa != pb) {
                        return pa < pb;
                    }

                    const auto & ea = cells.ext_get(a);

                    return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
                });

                rank.assign(n_kv, -1);

                for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                    rank[order[k]] = (int32_t) k;
                }

                ranked = true;

                group_cells();
            }

            GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");


            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                    if (grp_slots[g] != slots_full) {
                        continue;
                    }

                    grp_bid[g] = n_bid++;

                    bid_idx  .push_back((int32_t) (pb*r));
                    bid_cell .push_back(grp_first[g]);
                    bid_slot0.push_back(grp_slot0[g]);
                }
            }

            GGML_ASSERT(n_bid <= n_blocks);

            for (int32_t b = 0; b < n_bid; ++b) {
                int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

                if (ranked) {
                    const int32_t   c = bid_slot0[b];
                    const llama_pos p = cells.pos_get(c);
                    const auto &    e = cells.ext_get(c);

                    sec_pos[0] = p;
                    sec_pos[1] = e.y;
                    sec_pos[2] = e.x;
                    sec_pos[3] = p;
                }

                for (int64_t sec = 0; sec < 4 && dst_blk_pos; ++sec) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
                }
            }

            // unpooled cells all point at one spare block. a spare block exists only when some
            // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
            const bool     have_dead_g = n_bid < n_blocks;
            const int32_t  dead_bid_g  = have_dead_g ? n_bid : n_blocks - 1;

            if (tail_cells) {
                key_cell.assign(n_kv*key_seqs.size(), 0);
            }

            // the block -> cells map, and the per-sequence key -> cell map the tail reads
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];

                blk_of[j] = g < 0 ? -1 : grp_bid[g];

                if (g >= 0) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (blk_of[j] >= 0) {
                        cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
                    }

                    if (tail_cells && idx >= 0 && idx < n_kv) {
                        for (int64_t si = 0; si < (int64_t) key_seqs.size(); ++si) {
                            if (cells.seq_has((uint32_t) j, key_seqs[si])) {
                                key_cell[si*n_kv + idx] = (int32_t) j;
                            }
                        }
                    }
                }

                if (cur_cell_blk) {
                    cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid_g : blk_of[j];
                }
            }

        }

        const int32_t n_bid_last = n_bid;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // whole blocks: one value covers a block, and the caller adds the attention mask,
                // which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = b < n_bid_last && bid_idx[b] + r - 1 <= q &&
                            cells.seq_has((uint32_t) bid_cell[b], seq_id) ? 0.0f : -INFINITY;
                }

                if (!tail_cells) {
                    continue;   // per-cell selection: the caller walks the cells itself, no tail list
                }

                // Eq. 19: the cells of the query's own block up to and including the query. The map
                // is built per sequence, so a shared cells array cannot hand a query another
                // sequence's cells. Padding repeats the query's own cell, which is inert: the caller
                // only ever writes a 0 flag and the mask decides the outcome.
                int32_t * cur_tail = dst_tail + i*r;

                for (int64_t j = 0; j < r; ++j) {
                    const int64_t key = tail_start + j <= q ? tail_start + j : q;

                    int64_t cell;

                    if (fast) {
                        // the run is dense and single-sequence: slots are positions shifted by j0 - p0
                        cell = fast_j0 + (std::max<int64_t>(key, fast_p0) - fast_p0);
                    } else {
                        int64_t si = 0;

                        while (si < (int64_t) key_seqs.size() && key_seqs[si] != seq_id) {
                            ++si;
                        }

                        GGML_ASSERT(si < (int64_t) key_seqs.size());

                        const int64_t k0 = (int64_t) std::min<int64_t>(std::max<int64_t>(key, 0), n_kv - 1);
                        const int64_t k1 = (int64_t) std::min<int64_t>(std::max<int64_t>(q,   0), n_kv - 1);

                        cell = key_cell[si*n_kv + k0];

                        // a trimmed cache can lose the cell a key names; fall back to the query's own
                        if (!cells.seq_has((uint32_t) cell, seq_id)) {
                            cell = key_cell[si*n_kv + k1];
                        }
                    }

                    // never hand the caller an index outside the cache; a clamped cell is inert
                    cur_tail[j] = (int32_t) std::min<int64_t>(std::max<int64_t>(cell, 0), n_kv - 1);
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    // stamp the watermark only for the run the graph actually wrote: rows [0, pool.n_bid) hold valid
    // keys once that graph has run. if the run broke since the build, forget it and rebuild next time
    if (pool.mode != llama_qsa_pool::NONE) {
        if (run_fast) {
            qsa_runs[ratio] = { run_j0, run_p0, run_j1, run_b_lo, run_wm };
        } else {
            qsa_runs.erase(ratio);
        }

        LLAMA_LOG_DEBUG("%s: qsa pool: mode = %d, wm = %u, n_bid = %u, n_new = %u, written = %u, n_kv = %u\n", __func__,
                pool.mode, pool.wm, pool.n_bid, pool.n_new, run_wm, n_kv);
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())),
    is_update(true) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)),
    is_update(true) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();

        // update() copies whole stream buffers and applies the K-shift: the pool is positional, so it
        // cannot follow either
        if (is_update) {
            mem->qsa_pool_invalidate();
        }
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * blk_cells,
        ggml_tensor * tail_cells,
        ggml_tensor * cell_blk,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * new_rows,
        ggml_tensor * new_cells,
        ggml_tensor * new_pos,
        const llama_qsa_pool & pool,
        const llama_ubatch * ubatch,
        uint32_t n_kv,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(blk_cells, tail_cells, cell_blk, blk_pos, bias,
            new_rows, new_cells, new_pos, pool, ubatch, n_kv, ratio, blk_bias);
}

llama_qsa_pool llama_memory_hybrid_idx_context::qsa_pool_get(uint32_t ratio, const llama_ubatch & ubatch, uint32_t n_stream, uint32_t n_kv) const {
    GGML_ASSERT(mem != nullptr);

    // a full-cache or update context carries no ubatch state, so it asks for the worst case to reserve
    return mem->qsa_pool_get(ratio, ubatch, n_stream, n_kv, is_update);
}
