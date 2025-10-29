// vecadd.h
// C API for the vector-add CUDA helper. Exposes a simple function that takes
// host pointers (float*) and a length and performs device vector-add, copying
// the result back to the provided output pointer.

#pragma once

#ifdef _WIN32
#  ifdef VECADD_EXPORTS
#    define VECADD_API __declspec(dllexport)
#  else
#    define VECADD_API __declspec(dllimport)
#  endif
#else
#  define VECADD_API
#endif

extern "C" {
    // Returns 0 on success, non-zero on failure.
    VECADD_API int vecAdd_host(const float* a_host, const float* b_host, float* out_host, int n);
}
