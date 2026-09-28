#pragma once

#include "ggml.h"
#include "llama-mmap.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_lazy_reader {
    // path holds the table, whose row 0 starts at file offset offs
    llama_lazy_reader(const std::string & path, size_t offs, enum ggml_type type,
                      int64_t row_elems, int64_t n_rows, int n_readers);

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader();

    // fill dst with the n gathered rows, dequantized to F32; thread-safe
    void gather(const int32_t * rows, int64_t n, float * dst) const;

    // issue the WILLNEED calls of these rows on a background thread, so the reads run while the batch
    // that will use them computes; advisory, a wrong row list only wastes pages
    void prefetch(const int32_t * rows, int64_t n) const;

    int64_t n_rows()    const { return nrows; }
    int64_t row_elems() const { return relems; }
    size_t  row_size()  const { return rsize;  }
    int     n_readers() const { return (int) files.size(); }

private:
    // read the rows of pairs[begin, end) through files[fi], writing each to its slot
    void read_range(const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                    size_t fi, float * dst) const;

    // one WILLNEED per row; rows must be sorted and unique
    void fadvise_rows(const int32_t * rows, int64_t n) const;

    void prefetch_loop() const;

    // rows of the last prefetch, sorted and unique: gather skips its own WILLNEED loop when its own
    // row list matches, which is the case whenever the caller guessed the next window right
    mutable std::mutex              prefetch_mtx;
    mutable std::condition_variable prefetch_cv;
    mutable std::vector<int32_t>    prefetch_job;
    mutable std::vector<int32_t>    prefetch_done;
    mutable std::thread             prefetch_thread;
    mutable bool                    prefetch_pending = false;
    mutable bool                    prefetch_stop    = false;

    // one buffered file per reader thread: read_at is not thread-safe, and the loader's own descriptor may be direct I/O
    std::vector<std::unique_ptr<llama_file>> files;

    const size_t   offs;   // file offset of row 0
    const size_t   rsize;  // bytes per stored row
    const int64_t  relems; // F32 elements per row
    const int64_t  nrows;

    ggml_to_float_t to_float; // the dequantizer the ggml_get_rows CPU kernel uses; null for F32
};
