// throwaway harness: does the qsa block window break when the position line runs ahead of the cells?
//
// the block array is sized from the cell water mark (llama_kv_cache::get_n_kv pads used_max_p1() to
// 256) while blocks are cut on the position line, so the graph only stays in range while
// pos_max < ceil256(used_max_p1). an mrope draft cache violates it: draft-mtp skips embedding
// batches, so image cells never enter ctx_dft, but the positions it copies still carry the image's
// forward jump of max(nx, ny). a text batch with a gap in it is legal for an mrope model
// (llama-batch.cpp allows a forward jump), so the gap below reproduces the bench crash with no
// vision, no draft context and no weights bigger than a dummy.
//
// prediction: the crash is a function of the gap versus the padding slack, not of the depth.
// with n_prompt cells and a gap g, pos_max = n_prompt + g + i - 1 and cell_p1 = n_prompt + i, so it
// fires at the first chunk where ceil256(cell_p1) - cell_p1 <= g. g = 0 never fires, g = 64 fires
// partway through the run, g >= 256 fires on the first ubatch after the gap.
//
// the run is a script, so the same binary can measure the mrope duplicate case (an image pins one
// position across many cells) next to the jump case, and print a logits fingerprint for A/B:
//
//   build (from the repo root):
//     g++ -O1 -g3 -std=c++17 -Iinclude -Iggml/include \
//       .pi/agent/memory/experiments/tools/qsa-posgap-harness.cpp \
//       -o /tmp/qsa-posgap -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin
//   run:
//     Q4EXP_POOLED=<0|1> /tmp/qsa-posgap <model> <script> [n_chunk]
//     /tmp/qsa-posgap models/q4exp-48l-12qsa.gguf seq300,gap64,seq1024        // the field crash
//     /tmp/qsa-posgap models/q4exp-48l-12qsa.gguf seq300,pin64,gap64,seq1024  // + an image-like pin
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// seq<n>  n tokens at consecutive positions
// pin<n>  n tokens sharing one position, the way an mrope image pins its t line
// gap<n>  jump the position line forward by n, the way a draft cache skips the image cells
struct step {
    int kind;   // 0 = seq, 1 = pin, 2 = gap
    int n;
};

static std::vector<step> parse(const char * s) {
    std::vector<step> res;

    while (*s) {
        const char * e = strchr(s, ',');
        std::string tok(s, e ? (size_t) (e - s) : strlen(s));

        if (tok.rfind("seq", 0) == 0) {
            res.push_back({ 0, atoi(tok.c_str() + 3) });
        } else if (tok.rfind("pin", 0) == 0) {
            res.push_back({ 1, atoi(tok.c_str() + 3) });
        } else if (tok.rfind("gap", 0) == 0) {
            res.push_back({ 2, atoi(tok.c_str() + 3) });
        } else {
            fprintf(stderr, "bad step '%s'\n", tok.c_str());
            exit(2);
        }

        s += tok.size() + (e ? 1 : 0);
    }

    return res;
}

// the last token of each batch carries output, so the fingerprint below is the next-token dist
static void feed(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos, bool pin) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);

    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pin ? pos : pos + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = i + 1 == toks.size() ? 1 : 0;
    }

    b.n_tokens = (int) toks.size();

    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "FATAL decode failed at pos %d\n", (int) pos);
        exit(1);
    }

    llama_batch_free(b);
}

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "models/q4exp-48l-12qsa.gguf";
    const std::string scr  = argc > 2 ? argv[2] : "seq300,gap64,seq1024";
    const int n_chunk      = argc > 3 ? atoi(argv[3]) : 64;

    std::vector<step> script = parse(scr.c_str());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        return 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 8192;
    cp.n_batch   = n_chunk;
    cp.n_ubatch  = n_chunk;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 2;
    }

    const int       n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t  mem     = llama_get_memory(ctx);

    std::vector<llama_token> toks(16384);
    for (size_t i = 0; i < toks.size(); ++i) {
        toks[i] = (llama_token) ((i*7919 + 17) % n_vocab);
    }

    size_t  i_tok = 0;
    llama_pos pos = 0;
    int     cells = 0;

    fprintf(stderr, "MARK script = %s, chunk = %d\n", scr.c_str(), n_chunk);

    for (const step & st : script) {
        if (st.kind == 2) {
            pos += st.n;
            fprintf(stderr, "GAP  +%d -> pos %d (cells %d)\n", st.n, (int) pos, cells);
            continue;
        }

        for (int i = 0; i < st.n; i += n_chunk) {
            const int n = std::min(n_chunk, st.n - i);

            feed(ctx, std::vector<llama_token>(toks.begin() + i_tok, toks.begin() + i_tok + n), pos, st.kind == 1);

            i_tok += n;
            cells += n;

            // a pinned run leaves one position behind for n cells, the way an mrope image pins its t
            //     line; the forward jump that follows it is a separate gap step
            pos = st.kind == 1 ? pos + 1 : pos + n;

            fprintf(stderr, "STEP %s i=%4d cells~%5d pos_max=%5d\n", st.kind == 1 ? "pin" : "seq",
                    i, cells, (int) llama_memory_seq_pos_max(mem, 0));
        }
    }

    const float *    logits = llama_get_logits(ctx);
    const uint32_t * bits   = (const uint32_t *) logits;

    uint64_t ck = 1469598103934665603ull;
    for (int i = 0; i < n_vocab; ++i) {
        ck = (ck ^ bits[i]) * 1099511628211ull;
    }

    fprintf(stderr, "CKSUM %016llx pos_max=%d cells=%d\n", (unsigned long long) ck,
            (int) llama_memory_seq_pos_max(mem, 0), cells);

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
