// throwaway harness: does the qsa block key pool survive the rollback path this arch actually uses?
// llama.cpp checkpoints (and thus draft-mtp rejection) go through llama_state_seq_{get,set}_data_ext,
// because the recurrent cache refuses a partial seq_rm. flow: generate to n_cut, snapshot, generate
// more, restore the snapshot, replay, and compare token-for-token.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static llama_token greedy(llama_context * ctx, int i) {
    const float * logits = llama_get_logits_ith(ctx, i);
    if (!logits) { fprintf(stderr, "no logits at %d\n", i); exit(4); }

    const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    int best = 0;
    float bv = logits[0];

    for (int k = 1; k < n; ++k) {
        if (logits[k] > bv) {
            bv = logits[k];
            best = k;
        }
    }

    return best;
}

static void feed(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, bool logits) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);

    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = (logits && i + 1 == toks.size()) ? 1 : 0;
    }

    b.n_tokens = (int) toks.size();

    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }

    llama_batch_free(b);
}

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "models/q4exp-4l.gguf";
    const int n_prompt = argc > 2 ? atoi(argv[2]) : 300;
    const int n_cut    = argc > 3 ? atoi(argv[3]) : 250;
    const int n_gen    = argc > 4 ? atoi(argv[4]) : 600;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        return 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 8192;
    cp.n_batch   = 512;
    cp.n_ubatch  = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 2;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        prompt[i] = (llama_token) ((i*7919 + 17) % n_vocab);
    }

    std::vector<llama_token> full = prompt;

    feed(ctx, prompt, 0, true);

    // generate to the snapshot point
    for (int i = 0; i < n_cut; ++i) {
        llama_token t = greedy(ctx, (int) (i == 0 ? prompt.size() : 1) - 1);
        full.push_back(t);
        feed(ctx, std::vector<llama_token>(1, t), (llama_pos) (prompt.size() + i), true);
    }

    const size_t n_keep = full.size();

    std::vector<uint8_t> ckpt(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    const size_t n_ckpt = llama_state_seq_get_data_ext(ctx, ckpt.data(), ckpt.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

    fprintf(stderr, "MARK snapshot %zu bytes at %zu tokens\n", n_ckpt, n_keep);

    // a checkpoint save extracts and removes the sequence, so re-insert it to keep going
    if (llama_state_seq_set_data_ext(ctx, ckpt.data(), n_ckpt, 0, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
        fprintf(stderr, "restore failed\n");
        return 3;
    }

    // continue past the snapshot - the reference continuation
    std::vector<llama_token> cont = full;

    for (size_t i = n_keep; i < (size_t) n_gen; ++i) {
        llama_token t = greedy(ctx, 0);
        cont.push_back(t);
        feed(ctx, std::vector<llama_token>(1, t), (llama_pos) i, true);
    }

    // throw the current state away and put the snapshot back, then replay: a rollback
    std::vector<uint8_t> discard(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    llama_state_seq_get_data_ext(ctx, discard.data(), discard.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);

    if (llama_state_seq_set_data_ext(ctx, ckpt.data(), n_ckpt, 0, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
        fprintf(stderr, "restore 2 failed\n");
        return 3;
    }

    fprintf(stderr, "MARK replay\n");

    std::vector<llama_token> replay = cont;

    for (size_t i = n_keep; i < (size_t) n_gen; ++i) {
        llama_token t = greedy(ctx, 0);
        if (i < replay.size()) {
            replay[i] = t;
        } else {
            replay.push_back(t);
        }
        feed(ctx, std::vector<llama_token>(1, t), (llama_pos) i, true);
    }

    size_t diff = 0;
    for (size_t i = 0; i < cont.size() && i < replay.size(); ++i) {
        diff += cont[i] != replay[i];
    }

    unsigned long long s = 1469598103934665603ULL;
    for (auto t : cont) { s ^= (unsigned) t; s *= 1099511628211ULL; }

    printf("RESULT tokens=%zu rollback_replay_mismatch=%zu cksum=%llu\n",
           cont.size(), diff, s);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
