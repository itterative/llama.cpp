#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

// which pool variant a graph uses for the indexer's block keys, see plans/h9-pooled-block-keys.md
// the pool holds one key per complete block: pooled at the step the block finishes instead of being
// re-derived from the raw cache on every step
struct llama_qsa_pool {
    enum mode_e {
        NONE = 0,    // no pool for this graph: derive the block keys every step, write nothing
        CACHED,      // rows [0, wm) already sit in the pool: read them, write [wm, n_bid), score the result
    };

    mode_e   mode  = NONE;
    uint32_t wm    = 0;   // rows valid before this graph
    uint32_t n_bid = 0;   // complete blocks this ubatch scores
    uint32_t n_new = 0;   // rows this graph writes: [wm, n_bid)
};

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache. Blocks cut
    // the position line, not the cell array, so no caller assumes a contiguous layout:
    //   blk_cells I32 [ratio*n_blocks, ns]    cells making up each block, block-major
    //   tail_cells I32 [ratio, n_tokens/ns, ns] Eq. 19's per-query tail, resolved per sequence
    //   blk_pos   I32 [4*n_blocks*ns]         mrope position rows of each block's first token
    //   bias      F32 [n_blocks or n_kv, n_tokens/ns, ns] -inf where invisible, 0 where visible
    // blk_bias asks for the bias per block and selects whole blocks, so the tail is a separate index
    // list and cell_blk is not needed. Without blk_bias the selection runs per cell, bias is per cell,
    // and cell_blk I32 [n_kv, ns] (cell -> block) replaces tail_cells.
    // the attention mask is added afterwards, which is the only part of the bias that varies within a
    // block: padding or duplicate indices can only write a 0 flag, so they are inert
    //
    // pool carries the graph's choice for the pooled block keys (qsa_pool_get): the new_block* tensors
    // name the pool rows this ubatch writes, REBUILD and CACHED only. set_input_qsa stamps the
    // watermark, so it must run for every graph that is about to be computed.
    void set_input_qsa(ggml_tensor * blk_cells, ggml_tensor * tail_cells, ggml_tensor * cell_blk,
                       ggml_tensor * blk_pos, ggml_tensor * bias,
                       ggml_tensor * new_rows, ggml_tensor * new_cells, ggml_tensor * new_pos,
                       const llama_qsa_pool & pool,
                       const llama_ubatch * ubatch,
                       uint32_t n_kv, uint32_t ratio, bool blk_bias) const;

    // which pool variant a graph build should use for this ratio, see plans/h9-pooled-block-keys.md
    // pure query: no side effects, so it is safe from can_reuse and from graph reservation
    // n_kv is the cell window the graph sizes its tensors from, so the caller has to pass the same one
    // worst_case answers for a context with no ubatch state to test (graph reservation): the shape that
    // derives and writes every block, so that the reservation covers every pooled graph
    llama_qsa_pool qsa_pool_get(uint32_t ratio, const llama_ubatch & ubatch, uint32_t n_stream, uint32_t n_kv, bool worst_case) const;

    // the pooled keys are positional, so any mutation that can move cells or positions drops every
    // recorded run: the next fast-state build re-derives all complete blocks once and writes them back
    // const because qsa_runs is mutable: the batch contexts only hold the memory object as const
    void qsa_pool_invalidate() const {
        qsa_runs.clear();
    }

    // truncating at p0 leaves every position below it in place, so the blocks that end below it keep
    // their pooled keys. rolling a speculative draft back is a truncation, and clearing the whole run
    // for it would re-derive the entire cache on nearly every decode step
    void qsa_pool_truncate(llama_seq_id seq_id, llama_pos p0);

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // env: Q4EXP_POOLED, decides both the pool allocation and whether graphs may use it
    // declared first, so mem_idx below can see it while initialising
    bool qsa_pool_on = false;

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // one recorded run per compress ratio: rows [0, wm) of every layer's pool tensor are the block
    // keys of the single-sequence dense cell run identified by j0/p0/b_lo
    // mutable because the watermark is stamped from the const set_input path
    struct qsa_run {
        int32_t   j0    = -1;    // first cell of the run
        llama_pos p0    = -1;    // position of cell j0
        int32_t   j1    = -1;    // cells below this were verified dense and in position order
        uint32_t  b_lo  = 0;     // first complete block, = ceil(p0/ratio)
        uint32_t  wm    = 0;     // pooled rows known valid
    };

    mutable std::map<uint32_t, qsa_run> qsa_runs;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * blk_cells, ggml_tensor * tail_cells, ggml_tensor * cell_blk,
                       ggml_tensor * blk_pos, ggml_tensor * bias,
                       ggml_tensor * new_rows, ggml_tensor * new_cells, ggml_tensor * new_pos,
                       const llama_qsa_pool & pool, const llama_ubatch * ubatch,
                       uint32_t n_kv, uint32_t ratio, bool blk_bias) const;

    // see llama_memory_hybrid_idx::qsa_pool_get
    llama_qsa_pool qsa_pool_get(uint32_t ratio, const llama_ubatch & ubatch, uint32_t n_stream, uint32_t n_kv) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // update() moves whole stream buffers and rotates keys, which the pool cannot follow
    // true for the contexts built by init_update and init_full, false for batch contexts
    const bool is_update = false;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
