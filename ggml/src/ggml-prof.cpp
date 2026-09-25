#include "ggml-prof.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// region names are expected to be literals: the pointer is kept, not copied
#define GGML_PROF_MAX_REGIONS  64
#define GGML_PROF_MAX_COUNTERS 32
#define GGML_PROF_MAX_DEPTH    32

struct ggml_prof_region_stat {
    const char * name;
    uint64_t     total_ns;
    uint64_t     max_ns;
    uint64_t     min_ns;
    uint64_t     count;
};

static ggml_prof_region_stat ggml_prof_regions[GGML_PROF_MAX_REGIONS];
static int                   ggml_prof_n_regions = 0;

struct ggml_prof_counter_stat {
    const char * name;
    uint64_t     total;
    uint64_t     count;
};

static ggml_prof_counter_stat ggml_prof_counters[GGML_PROF_MAX_COUNTERS];
static int                    ggml_prof_n_counters = 0;

static const struct ggml_prof_sink * ggml_prof_sink = nullptr;

// counters are not locked: a multi-threaded run may lose or double-count time. mirrored ranges are exact
static thread_local struct {
    int      idx[GGML_PROF_MAX_DEPTH];
    uint64_t t0[GGML_PROF_MAX_DEPTH];
    int      depth;
} ggml_prof_thread;

static void ggml_prof_report_atexit(void);

void ggml_prof_register_sink(const struct ggml_prof_sink * sink) {
    ggml_prof_sink = sink;
}

static int ggml_prof_region_idx(const char * name) {
    for (int i = 0; i < ggml_prof_n_regions; ++i) {
        if (ggml_prof_regions[i].name == name || strcmp(ggml_prof_regions[i].name, name) == 0) {
            return i;
        }
    }

    if (ggml_prof_n_regions >= GGML_PROF_MAX_REGIONS) {
        return -1;
    }

    const int i = ggml_prof_n_regions++;

    ggml_prof_regions[i] = { name, 0, 0, UINT64_MAX, 0 };

    return i;
}

static bool ggml_prof_regions_enabled(void) {
    static const bool enabled = []() {
        const char * env = getenv("GGML_PROF_REGIONS");

        if (!env || atoi(env) == 0) {
            return false;
        }

        fprintf(stderr, "[prof] regions on: counters%s%s\n", ggml_prof_sink && ggml_prof_sink->region_begin ? ", sink '" : "",
                ggml_prof_sink && ggml_prof_sink->region_begin ? ggml_prof_sink->name : "");

        atexit(ggml_prof_report_atexit);

        return true;
    }();

    return enabled;
}

int ggml_prof_enabled(void) {
    return ggml_prof_regions_enabled() ? 1 : 0;
}

static int ggml_prof_counter_idx(const char * name) {
    for (int i = 0; i < ggml_prof_n_counters; ++i) {
        if (ggml_prof_counters[i].name == name || strcmp(ggml_prof_counters[i].name, name) == 0) {
            return i;
        }
    }

    if (ggml_prof_n_counters >= GGML_PROF_MAX_COUNTERS) {
        return -1;
    }

    const int i = ggml_prof_n_counters++;

    ggml_prof_counters[i] = { name, 0, 0 };

    return i;
}

void ggml_prof_count(const char * name, uint64_t delta) {
    if (!ggml_prof_regions_enabled()) {
        return;
    }

    const int idx = ggml_prof_counter_idx(name);

    if (idx < 0) {
        return;
    }

    ggml_prof_counters[idx].total += delta;
    ggml_prof_counters[idx].count += 1;
}

void ggml_prof_region_begin(const char * name) {
    if (!ggml_prof_regions_enabled()) {
        return;
    }

    int idx = -1;

    if (ggml_prof_thread.depth < GGML_PROF_MAX_DEPTH) {
        idx = ggml_prof_region_idx(name);

        if (idx >= 0 && ggml_prof_sink && ggml_prof_sink->region_begin) {
            ggml_prof_sink->region_begin(name);
        }
    }

    ggml_prof_thread.idx[ggml_prof_thread.depth] = idx;
    ggml_prof_thread.t0[ggml_prof_thread.depth]  = (uint64_t) ggml_time_us() * 1000;
    ggml_prof_thread.depth++;
}

void ggml_prof_region_end(void) {
    if (!ggml_prof_regions_enabled()) {
        return;
    }

    if (ggml_prof_thread.depth == 0) {
        return;
    }

    const int      i   = --ggml_prof_thread.depth;
    const int      idx = ggml_prof_thread.idx[i];
    const uint64_t t1  = (uint64_t) ggml_time_us() * 1000;

    if (idx < 0) {
        return;
    }

    ggml_prof_region_stat * r  = &ggml_prof_regions[idx];
    const uint64_t          dt = t1 - ggml_prof_thread.t0[i];

    r->total_ns += dt;
    r->count    += 1;
    r->max_ns    = dt > r->max_ns ? dt : r->max_ns;
    r->min_ns    = dt < r->min_ns ? dt : r->min_ns;

    if (ggml_prof_sink && ggml_prof_sink->region_end) {
        ggml_prof_sink->region_end();
    }
}

// unit agnostic magnitude, so byte counters stay readable
static void ggml_prof_fmt(uint64_t v, char * buf, size_t len) {
    if      (v >= 1000000000ULL) { snprintf(buf, len, "%.2fG", v*1e-9); }
    else if (v >= 1000000ULL)    { snprintf(buf, len, "%.2fM", v*1e-6); }
    else if (v >= 1000ULL)       { snprintf(buf, len, "%.2fk", v*1e-3); }
    else                         { snprintf(buf, len, "%llu", (unsigned long long) v); }
}

void ggml_prof_report(const char * title) {
    fprintf(stderr, "[prof] %s: %d regions\n", title ? title : "regions", ggml_prof_n_regions);

    // registration order is not interesting, so sort by accumulated time
    const ggml_prof_region_stat * order[GGML_PROF_MAX_REGIONS];
    int                           ord_n = 0;

    for (int i = 0; i < ggml_prof_n_regions; ++i) {
        int j = ord_n++;

        while (j > 0 && order[j-1]->total_ns < ggml_prof_regions[i].total_ns) {
            order[j] = order[j-1];
            j--;
        }

        order[j] = &ggml_prof_regions[i];
    }

    for (int i = 0; i < ord_n; ++i) {
        const ggml_prof_region_stat * r = order[i];

        fprintf(stderr, "[prof]   %-28s %8llu calls %10.2f ms total %8.4f ms/call %6.2f..%8.2f ms\n",
                r->name,
                (unsigned long long) r->count,
                r->total_ns * 1e-6,
                r->count ? r->total_ns * 1e-6 / r->count : 0.0,
                r->count ? r->min_ns * 1e-6 : 0.0,
                r->max_ns  * 1e-6);
    }

    for (int i = 0; i < ggml_prof_n_regions; ++i) {
        ggml_prof_regions[i] = { ggml_prof_regions[i].name, 0, 0, UINT64_MAX, 0 };
    }

    if (ggml_prof_n_counters > 0) {
        fprintf(stderr, "[prof] counters\n");

        // name order, so a counter and the one it gets divided by stay adjacent
        const ggml_prof_counter_stat * corder[GGML_PROF_MAX_COUNTERS];
        int                            cord_n = 0;

        for (int i = 0; i < ggml_prof_n_counters; ++i) {
            int j = cord_n++;

            while (j > 0 && strcmp(corder[j-1]->name, ggml_prof_counters[i].name) > 0) {
                corder[j] = corder[j-1];
                j--;
            }

            corder[j] = &ggml_prof_counters[i];
        }

        for (int i = 0; i < cord_n; ++i) {
            const ggml_prof_counter_stat * c = corder[i];

            char tot[16];
            char avg[16];

            ggml_prof_fmt(c->total, tot, sizeof(tot));
            ggml_prof_fmt(c->count ? c->total / c->count : 0, avg, sizeof(avg));

            fprintf(stderr, "[prof]   %-28s %8llu calls %12s total %10s/call\n",
                    c->name, (unsigned long long) c->count, tot, avg);
        }
    }

    for (int i = 0; i < ggml_prof_n_counters; ++i) {
        ggml_prof_counters[i] = { ggml_prof_counters[i].name, 0, 0 };
    }
}

static void ggml_prof_report_atexit(void) {
    ggml_prof_report("exit");
}

// the capture window is independent of the counters: it only does something when a profiler is attached
void ggml_prof_window_begin(void) {
    if (ggml_prof_sink && ggml_prof_sink->window_begin) {
        ggml_prof_sink->window_begin();
    }
}

void ggml_prof_window_end(void) {
    if (ggml_prof_sink && ggml_prof_sink->window_end) {
        ggml_prof_sink->window_end();
    }
}
