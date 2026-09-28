#include "llama-lazy-reader.h"

#include "llama-impl.h"
#include "ggml-prof.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

#ifdef __has_include
    #if __has_include(<fcntl.h>)
        #include <fcntl.h>
    #endif
    #if __has_include(<unistd.h>)
        #include <unistd.h>
    #endif
#endif

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

llama_lazy_reader::~llama_lazy_reader() {
    {
        std::lock_guard<std::mutex> lock(prefetch_mtx);

        prefetch_stop = true;
    }

    prefetch_cv.notify_one();

    if (prefetch_thread.joinable()) {
        prefetch_thread.join();
    }
}

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
// storage served, so the pair gives the miss rate. reading this file counts itself in rchar, so the
// caller gets the length back and subtracts it
static bool lazy_self_io(uint64_t & rchar, uint64_t & storage, size_t & n_read) {
#if defined(__linux__)
    static const int fd = open("/proc/self/io", O_RDONLY);

    if (fd < 0) {
        return false;
    }

    char buf[256];
    const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);

    if (n <= 0) {
        return false;
    }

    n_read = (size_t) n;
    buf[n] = '\0';

    bool got_rchar   = false;
    bool got_storage = false;

    for (char * line = buf; line != nullptr; ) {
        char * nl = strchr(line, '\n');

        if (nl != nullptr) {
            *nl = '\0';
        }

        if (strncmp(line, "rchar:", 6) == 0) {
            rchar = strtoull(line + 6, nullptr, 10);
            got_rchar = true;
        } else if (strncmp(line, "read_bytes:", 11) == 0) {
            storage = strtoull(line + 11, nullptr, 10);
            got_storage = true;
        }

        line = nl != nullptr ? nl + 1 : nullptr;
    }

    return got_rchar && got_storage;
#else
    (void) rchar;
    (void) storage;
    (void) n_read;

    return false;
#endif
}

// a gather of at most this many rows is decode or a speculative verify, above it is prefill
static const int64_t LAZY_IO_DECODE_MAX_ROWS = 1024;

// env: LLAMA_LAZY_PREFETCH - ask for every distinct row before waiting on any of them. on by default:
// the rows are scattered, so reading them serially makes each cold one a separate queue-depth-1 wait.
// set to 0 to skip it, which is slightly cheaper when the table is small enough to stay cached
static int llama_lazy_prefetch() {
    static const int val = []() {
        const char * env = std::getenv("LLAMA_LAZY_PREFETCH");

        return env == nullptr || atoi(env) != 0 ? 1 : 0;
    }();

    return val;
}

// env: LLAMA_LAZY_WORKERS - reader threads per gather. 0 keeps the default of one per 32 rows, which
// leaves a decode gather of 16 rows on a single thread
static int llama_lazy_workers() {
    static const int val = []() {
        const char * env = std::getenv("LLAMA_LAZY_WORKERS");

        return env != nullptr ? std::max(0, atoi(env)) : 0;
    }();

    return val;
}

// env: LLAMA_LAZY_PREFETCH_AHEAD - issue the next batch's WILLNEED calls from a background thread while
// the current batch computes. on by default; 0 keeps them inline in gather() only
static int llama_lazy_prefetch_ahead() {
    static const int val = []() {
        const char * env = std::getenv("LLAMA_LAZY_PREFETCH_AHEAD");

        return env == nullptr || atoi(env) != 0 ? 1 : 0;
    }();

    return val;
}

// one WILLNEED per row; rows must be sorted and unique. the kernel rounds each call up to a page, which
// is the granularity the table layout leaves us - the rows of one token are 16 scattered 110 B spans
void llama_lazy_reader::fadvise_rows(const int32_t * rows, int64_t n) const {
#ifdef POSIX_FADV_WILLNEED
    const int fd = files[0]->file_id();

    for (int64_t i = 0; i < n; ++i) {
        posix_fadvise(fd, offs + (size_t) rows[i] * rsize, rsize, POSIX_FADV_WILLNEED);
    }
#else
    (void) rows;
    (void) n;
#endif
}

void llama_lazy_reader::prefetch(const int32_t * rows, int64_t n) const {
    if (n <= 0 || llama_lazy_prefetch() == 0 || llama_lazy_prefetch_ahead() == 0) {
        return;
    }

    std::vector<int32_t> job(rows, rows + n);
    std::sort(job.begin(), job.end());
    job.erase(std::unique(job.begin(), job.end()), job.end());

    {
        std::lock_guard<std::mutex> lock(prefetch_mtx);

        if (!prefetch_thread.joinable()) {
            prefetch_thread = std::thread([this]() { prefetch_loop(); });
        }

        // a job still waiting is stale by construction: it holds the window before this one
        prefetch_job.swap(job);
        prefetch_pending = true;
    }

    prefetch_cv.notify_one();

    // counters are not thread-safe, so the background thread never touches them
    ggml_prof_count("io:prefetch_rows", (uint64_t) n);
}

void llama_lazy_reader::prefetch_loop() const {
    for (;;) {
        std::vector<int32_t> job;

        {
            std::unique_lock<std::mutex> lock(prefetch_mtx);

            prefetch_cv.wait(lock, [this]() { return prefetch_stop || prefetch_pending; });

            if (prefetch_stop) {
                return;
            }

            job.swap(prefetch_job);
            prefetch_pending = false;
        }

        fadvise_rows(job.data(), (int64_t) job.size());

        {
            std::lock_guard<std::mutex> lock(prefetch_mtx);

            prefetch_done.swap(job);
        }
    }
}

// rows an earlier gather already read, so the reuse rate falls out of the counters. process-wide and
// never evicted, which is fine because it is inert unless profiling and a run touches few rows
static std::mutex                  lazy_seen_lock;
static std::unordered_set<int32_t> lazy_seen;

void llama_lazy_reader::gather(const int32_t * rows, int64_t n, float * dst) const {
    const bool prof = ggml_prof_enabled() != 0;

    uint64_t   rchar0 = 0, storage0 = 0;
    size_t     n_read0 = 0;
    const bool io = prof && lazy_self_io(rchar0, storage0, n_read0);

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

    // the distinct rows, sorted, which is also the list a prefetch one batch ahead would have issued
    std::vector<int32_t> distinct;
    distinct.reserve(n);

    for (int64_t i = 0; i < n; ) {
        int64_t j = i + 1;
        while (j < n && pairs[j].first == pairs[i].first) {
            ++j;
        }

        distinct.push_back(pairs[i].first);

        i = j;
    }

    // pairs is sorted, so the rows an earlier gather saw are one linear pass over the same distinct list
    int64_t n_uniq = (int64_t) distinct.size(), n_reuse = 0;

    if (prof) {
        std::lock_guard<std::mutex> lock(lazy_seen_lock);

        for (const int32_t row : distinct) {
            if (!lazy_seen.insert(row).second) {
                n_reuse++;
            }
        }
    }

    // a row list the prefetch already covered is resident or in flight, so the WILLNEED calls are not
    // just redundant, they are the expensive part: 4.3 us cold against 0.75 us cached, per row
    bool ahead = false;

    if (llama_lazy_prefetch() != 0 && llama_lazy_prefetch_ahead() != 0) {
        std::lock_guard<std::mutex> lock(prefetch_mtx);

        ahead = prefetch_done == distinct;
    }

    if (llama_lazy_prefetch() != 0 && !ahead) {
        // the region is opened here and not in fadvise_rows: the prefetch thread must not touch the
        // prof tables, which are plain globals with no lock
        ggml_prof_region prof_prefetch("lazy:prefetch");

        fadvise_rows(distinct.data(), (int64_t) distinct.size());
    }

    int64_t n_want = n / 32;
    if (llama_lazy_workers() > 0) {
        n_want = llama_lazy_workers();
    }

    const int n_workers = (int) std::min<int64_t>(files.size(), std::clamp<int64_t>(n_want, 1, std::max<int64_t>(1, n)));

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

    if (prof) {
        const bool dec = n <= LAZY_IO_DECODE_MAX_ROWS;

        ggml_prof_count(dec ? "io:rows_decode"  : "io:rows_prefill",  (uint64_t) n);
        ggml_prof_count(dec ? "io:uniq_decode"  : "io:uniq_prefill",  (uint64_t) n_uniq);
        ggml_prof_count(dec ? "io:reuse_decode" : "io:reuse_prefill", (uint64_t) n_reuse);

        if (llama_lazy_prefetch() != 0 && llama_lazy_prefetch_ahead() != 0) {
            // did the window issued one batch ahead match the rows this gather asked for
            ggml_prof_count(dec ? "io:ahead_hit_decode"  : "io:ahead_hit_prefill",  (uint64_t) ahead);
            ggml_prof_count(dec ? "io:ahead_miss_decode" : "io:ahead_miss_prefill", (uint64_t) !ahead);
        }

        if (io) {
            uint64_t rchar1 = 0, storage1 = 0;
            size_t   n_read1 = 0;

            if (lazy_self_io(rchar1, storage1, n_read1)) {
                // only the first read lands in the window: the second is charged after it reports
                ggml_prof_count(dec ? "io:rchar_decode"   : "io:rchar_prefill",   rchar1 - rchar0 - n_read0);
                ggml_prof_count(dec ? "io:storage_decode" : "io:storage_prefill", storage1 - storage0);
            }
        }
    }

    for (const auto & err : errs) {
        if (err) {
            std::rethrow_exception(err);
        }
    }
}
