// roctx sink for the ggml_prof region API (see ggml-prof.h).
//
// Loaded by name only: no link-time dependency on the profiler. Where the library is absent the calls
// stay no-ops and the region counters in ggml-base are the only effect.
//
// The library is resolved lazily, on every call, until it is found. Resolving it at load time does not
// work under rocprofv3: the app's shared objects are initialised while the tool's own SDK is still
// starting up, so the shim's dependencies are not yet there and the dlopen fails permanently.

#include "ggml-backend-impl.h"

#if defined(GGML_USE_HIP) && !defined(_WIN32)

#include <dlfcn.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void * ggml_roctx_open_one(const char * name) {
    void * h = dlopen(name, RTLD_NOW | RTLD_GLOBAL);

    if (h && !dlsym(h, "roctxRangePushA")) {
        dlclose(h);

        return nullptr;
    }

    return h;
}

// the directory the HIP runtime was loaded from, so a non-standard ROCm prefix needs no hard-coding
static void * ggml_roctx_open_prefix(void) {
    void * hip = dlsym(RTLD_DEFAULT, "hipGetDevice");

    Dl_info info;

    if (!hip || !dladdr(hip, &info) || !info.dli_fname) {
        return nullptr;
    }

    char path[PATH_MAX];
    const char * slash = strrchr(info.dli_fname, '/');

    if (!slash || (size_t) (slash - info.dli_fname) >= sizeof(path) - 40) {
        return nullptr;
    }

    memcpy(path, info.dli_fname, slash - info.dli_fname);
    path[slash - info.dli_fname] = '\0';

    for (const char * suffix : { "/librocprofiler-sdk-roctx.so", "/librocprofiler-sdk-roctx.so.1", "/librocprofiler-sdk-roctx.so.0" }) {
        strcat(path, suffix);

        void * h = ggml_roctx_open_one(path);

        path[slash - info.dli_fname] = '\0';

        if (h) {
            return h;
        }
    }

    return nullptr;
}

struct ggml_roctx_api {
    void (*range_push)(const char *);
    int32_t (*range_pop)(void);
    int32_t (*resume)(uint64_t);
    int32_t (*pause)(uint64_t);
};

static ggml_roctx_api ggml_roctx;
static int  ggml_roctx_tries            = 0;
static bool ggml_roctx_logged_missing   = false;
static bool ggml_roctx_logged_attach    = false;

template <typename T>
static T ggml_roctx_sym(void * h, const char * name) {
    return reinterpret_cast<T>(dlsym(h, name));
}

static bool ggml_roctx_tracing(void) {
    const char * env = getenv("GGML_PROF_REGIONS");

    return env && atoi(env);
}

// retried a few times, not forever: under a profiler the first regions can be reached before the shim
// is ready, but a box without the SDK must not pay a failed dlopen on every region. window_begin passes
// late=true, since opening a capture window is rare and happens well after load
static void ggml_roctx_resolve(bool late) {
    if (ggml_roctx.range_push) {
        return;
    }

    if (!late && ++ggml_roctx_tries > 16) {
        return;
    }

    // under rocprofv3 the shim can already be in the global scope, without a loadable path
    void * h = dlsym(RTLD_DEFAULT, "roctxRangePushA") ? (void *) RTLD_DEFAULT : nullptr;

    for (const char * name : { "librocprofiler-sdk-roctx.so", "librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so.0" }) {
        if (h) {
            break;
        }

        h = ggml_roctx_open_one(name);
    }

    if (!h) {
        h = ggml_roctx_open_prefix();
    }

    if (!h) {
        if (ggml_roctx_tracing() && !ggml_roctx_logged_missing) {
            ggml_roctx_logged_missing = true;

            fprintf(stderr, "[prof] roctx: library not found, counters only\n");
        }

        return;
    }

    ggml_roctx.range_push = ggml_roctx_sym<void (*)(const char *)>(h, "roctxRangePushA");
    ggml_roctx.range_pop  = ggml_roctx_sym<int32_t (*)(void)>(h, "roctxRangePop");
    ggml_roctx.resume     = ggml_roctx_sym<int32_t (*)(uint64_t)>(h, "roctxProfilerResume");
    ggml_roctx.pause      = ggml_roctx_sym<int32_t (*)(uint64_t)>(h, "roctxProfilerPause");

    if (ggml_roctx_tracing() && !ggml_roctx_logged_attach) {
        ggml_roctx_logged_attach = true;

        fprintf(stderr, "[prof] roctx: attached, roctxProfilerResume %s (without it, --selected-regions records nothing)\n",
                ggml_roctx.resume ? "present" : "missing");
    }
}

static void ggml_roctx_region_begin(const char * name) {
    ggml_roctx_resolve(false);

    if (ggml_roctx.range_push) {
        ggml_roctx.range_push(name);
    }
}

static void ggml_roctx_region_end(void) {
    if (ggml_roctx.range_pop) {
        ggml_roctx.range_pop();
    }
}

// the resume/pause pair delimits a capture window for rocprofv3 --selected-regions
static void ggml_roctx_window_begin(void) {
    ggml_roctx_resolve(true);

    if (ggml_roctx.resume) {
        ggml_roctx.resume(0);
    }
}

static void ggml_roctx_window_end(void) {
    if (ggml_roctx.pause) {
        ggml_roctx.pause(0);
    }
}

static const struct ggml_prof_sink ggml_prof_roctx_sink = {
    /* .name         = */ "roctx",
    /* .region_begin = */ ggml_roctx_region_begin,
    /* .region_end   = */ ggml_roctx_region_end,
    /* .window_begin = */ ggml_roctx_window_begin,
    /* .window_end   = */ ggml_roctx_window_end,
};

// registered unconditionally, since availability is only knowable at first use
static const bool ggml_prof_roctx_registered = []() {
    ggml_prof_register_sink(&ggml_prof_roctx_sink);

    return true;
}();

#else

// keep the translation unit non-empty on CUDA-only and Windows builds
static const int ggml_prof_roctx_unavailable = 0;

#endif
