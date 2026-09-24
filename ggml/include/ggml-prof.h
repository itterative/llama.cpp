#pragma once

#include "ggml.h"

// Named region timing for host-side phases of a run.
//
// Two uses of the same annotations:
//   - counters: when GGML_PROF_REGIONS is set, each region accumulates count / total / max wall ms and
//     the table is printed at exit. Works on any backend.
//   - profiler ranges: a backend may register a ggml_prof_sink (see ggml-backend-impl.h) that mirrors
//     each region into its own profiler, so a trace groups kernel rows by phase. The HIP backend does
//     this for rocprofv3; other builds simply keep the counters.
//
// Off by default: with the env var unset the pair of calls is two loads and a branch.
//
// Regions nest, per thread. End matches the most recent begin on that thread. Totals are inclusive of
// nested regions, so do not sum them; subtract to get self time.

#ifdef  __cplusplus
extern "C"
{
#endif

    // 0 when disabled, so a caller can skip building a region name at runtime
    GGML_API int ggml_prof_enabled(void);

    GGML_API void ggml_prof_region_begin(const char * name);
    GGML_API void ggml_prof_region_end(void);

    // print the accumulated table, then reset it
    GGML_API void ggml_prof_report(const char * title);

    // start / stop a whole-run capture window (rocprofv3 --selected-regions); no-op without the SDK
    GGML_API void ggml_prof_window_begin(void);
    GGML_API void ggml_prof_window_end(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

struct ggml_prof_region {
    ggml_prof_region(const char * name) {
        ggml_prof_region_begin(name);
    }

    ~ggml_prof_region() {
        ggml_prof_region_end();
    }

    ggml_prof_region(const ggml_prof_region &)            = delete;
    ggml_prof_region & operator=(const ggml_prof_region &) = delete;
};

struct ggml_prof_window {
    ggml_prof_window() {
        ggml_prof_window_begin();
    }

    ~ggml_prof_window() {
        ggml_prof_window_end();
    }

    ggml_prof_window(const ggml_prof_window &)            = delete;
    ggml_prof_window & operator=(const ggml_prof_window &) = delete;
};

#endif
