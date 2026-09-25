#include "llama-lazy-reader.h"

#include "llama-impl.h"
#include "ggml-prof.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>

llama_lazy_reader::llama_lazy_reader(const std::string & path, size_t offs, enum ggml_type type,
                                     int64_t row_elems, int64_t n_rows, int n_readers) :
    offs(offs),
    rsize(ggml_row_size(type, row_elems)),
    relems(row_elems),
    nrows(n_rows),
    to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
    if (type != GGML_TYPE_F32 && to_float == nullptr) {
        throw std::runtime_error(format("%s cannot be read row by row: %s has no F32 conversion",
                path.c_str(), ggml_type_name(type)));
    }

    GGML_ASSERT(row_elems > 0 && n_rows > 0 && n_readers > 0);

    files.reserve(n_readers);
    for (int i = 0; i < n_readers; ++i) {
        files.emplace_back(std::make_unique<llama_file>(path.c_str(), "rb", /*use_direct_io =*/ false));
    }
}

llama_lazy_reader::~llama_lazy_reader() = default;

void llama_lazy_reader::read_range(const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                                   size_t fi, float * dst) const {
    std::vector<uint8_t> bounce(rsize);

    for (int64_t i = begin; i < end; ) {
        int64_t j = i;
        while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
            ++j;
        }

        files[fi]->read_at(offs + (size_t) pairs[i].first * rsize, bounce.data(), rsize);

        float * first = dst + (size_t) pairs[i].second * relems;
        if (to_float) {
            to_float(bounce.data(), first, relems);
        } else {
            memcpy(first, bounce.data(), (size_t) relems * sizeof(float));
        }

        for (int64_t k = i + 1; k <= j; ++k) {
            memcpy(dst + (size_t) pairs[k].second * relems, first, (size_t) relems * sizeof(float));
        }

        i = j + 1;
    }
}

// rchar is every byte asked of read()/pread(), page cache hits included; read_bytes is only what
// storage served, so the pair gives the miss rate. these two reads add ~400 B to rchar themselves
static bool lazy_self_io(uint64_t & rchar, uint64_t & storage) {
    std::ifstream f("/proc/self/io");

    if (!f) {
        return false;
    }

    std::string key;
    uint64_t    val;
    bool        got_rchar = false, got_storage = false;

    while (f >> key >> val) {
        if      (key == "rchar:")      { rchar = val;   got_rchar = true;   }
        else if (key == "read_bytes:") { storage = val; got_storage = true; }
    }

    return got_rchar && got_storage;
}

// a gather of at most this many rows is decode or a speculative verify, above it is prefill
static const int64_t LAZY_IO_DECODE_MAX_ROWS = 1024;

void llama_lazy_reader::gather(const int32_t * rows, int64_t n, float * dst) const {
    uint64_t     rchar0 = 0, storage0 = 0;
    const bool   io = ggml_prof_enabled() && lazy_self_io(rchar0, storage0);

    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < nrows);
        pairs.emplace_back(rows[i], (int32_t) i);
    }

    {
        ggml_prof_region prof_sort("lazy:sort");

        std::sort(pairs.begin(), pairs.end());
    }

    const int n_workers = (int) std::min<int64_t>(files.size(), std::max<int64_t>(1, n / 32));

    auto run_chunk = [&](int w, std::exception_ptr & err) {
        try {
            read_range(pairs.data(), n * w / n_workers, n * (w + 1) / n_workers, w, dst);
        } catch (...) {
            err = std::current_exception();
        }
    };

    std::vector<std::exception_ptr> errs(n_workers);
    std::vector<std::thread> workers;
    try {
        workers.reserve(n_workers - 1);
        for (int w = 1; w < n_workers; ++w) {
            workers.emplace_back([&run_chunk, &errs, w]() { run_chunk(w, errs[w]); });
        }
    } catch (...) {
        for (auto & t : workers) {
            t.join();
        }
        throw;
    }

    run_chunk(0, errs[0]);

    for (auto & t : workers) {
        t.join();
    }

    if (io) {
        uint64_t rchar1 = 0, storage1 = 0;

        if (lazy_self_io(rchar1, storage1)) {
            const bool dec = n <= LAZY_IO_DECODE_MAX_ROWS;

            ggml_prof_count(dec ? "io:rows_decode"    : "io:rows_prefill",    (uint64_t) n);
            ggml_prof_count(dec ? "io:rchar_decode"   : "io:rchar_prefill",   rchar1 - rchar0);
            ggml_prof_count(dec ? "io:storage_decode" : "io:storage_prefill", storage1 - storage0);
        }
    }

    for (const auto & err : errs) {
        if (err) {
            std::rethrow_exception(err);
        }
    }
}
