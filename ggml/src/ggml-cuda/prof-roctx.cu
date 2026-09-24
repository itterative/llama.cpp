// roctx sink for the ggml_prof region API (see ggml-prof.h).
//
// Loaded by name only: no link-time dependency on the profiler, and where the library is absent the
// sink is simply never registered, leaving the region counters in ggml-base as the only effect.
// The handle is deliberately never closed, so the cached symbols stay valid through exit.

#include "ggml-backend-impl.h"

#if defined(GGML_USE_HIP) && !defined(_WIN32)

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

static void * ggml_roctx_lib(void) {
    static void * handle = []() {
        // under rocprofv3 the shim can already be in the global scope, without a loadable path
        if (dlsym(RTLD_DEFAULT, "roctxRangePushA")) {
            return (void *) RTLD_DEFAULT;
        }

        const char * candidates[] = {
            "librocprofiler-sdk-roctx.so",
            "librocprofiler-sdk-roctx.so.3",
            "librocprofiler-sdk-roctx.so.2",
            "librocprofiler-sdk-roctx.so.1",
            "librocprofiler-sdk-roctx.so.0",
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so.3",
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so.2",
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so.1",
            "/opt/rocm/lib/librocprofiler-sdk-roctx.so.0",
        };

        for (const char * c : candidates) {
            void * h = dlopen(c, RTLD_NOW | RTLD_LOCAL);

            if (h && dlsym(h, "roctxRangePushA")) {
                return h;
            }
        }

        return (void *) nullptr;
    }();

    return handle;
}

template <typename T>
static T ggml_roctx_sym(const char * name) {
    return reinterpret_cast<T>(dlsym(ggml_roctx_lib(), name));
}

static void ggml_roctx_region_begin(const char * name) {
    static auto fn = ggml_roctx_sym<void(*)(const char *)>("roctxRangePushA");

    if (fn) {
        fn(name);
    }
}

static void ggml_roctx_region_end(void) {
    static auto fn = ggml_roctx_sym<int32_t(*)(void)>("roctxRangePop");

    if (fn) {
        fn();
    }
}

// the resume/pause pair delimits a capture window for rocprofv3 --selected-regions
static void ggml_roctx_window_begin(void) {
    static auto fn = ggml_roctx_sym<int32_t(*)(uint64_t)>("roctxProfilerResume");

    if (fn) {
        fn(0);
    }
}

static void ggml_roctx_window_end(void) {
    static auto fn = ggml_roctx_sym<int32_t(*)(uint64_t)>("roctxProfilerPause");

    if (fn) {
        fn(0);
    }
}

static const struct ggml_prof_sink ggml_prof_roctx_sink = {
    /* .name         = */ "roctx",
    /* .region_begin = */ ggml_roctx_region_begin,
    /* .region_end   = */ ggml_roctx_region_end,
    /* .window_begin = */ ggml_roctx_window_begin,
    /* .window_end   = */ ggml_roctx_window_end,
};

// registering at load keeps the call sites free of any backend-specific code
static const bool ggml_prof_roctx_registered = []() {
    void * handle = ggml_roctx_lib();

    if (handle) {
        ggml_prof_register_sink(&ggml_prof_roctx_sink);
    }

    // which half of the handshake is missing is otherwise invisible: without the library there are no
    // ranges and no windows, and without roctxProfilerResume the ranges work but --selected-regions
    // records nothing
    const char * env = getenv("GGML_PROF_REGIONS");

    if (env && atoi(env)) {
        fprintf(stderr, "[prof] roctx: lib %s, roctxRangePushA %s, roctxProfilerResume %s\n",
                handle == nullptr                          ? "not found"
                    : handle == RTLD_DEFAULT               ? "already loaded" : "loaded",
                handle ? (dlsym(handle, "roctxRangePushA")       ? "present" : "missing") : "-",
                handle ? (dlsym(handle, "roctxProfilerResume")   ? "present" : "missing") : "-");
    }

    return handle != nullptr;
}();

#else

// keep the translation unit non-empty on CUDA-only and Windows builds
static const int ggml_prof_roctx_unavailable = 0;

#endif
