// vecadd_lib.cu
// Exports a simple C-callable API to perform vector addition on the GPU.

#include <cuda_runtime.h>
#include <cstdio>
#include "../include/vecadd.h"

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    }
}

__global__ void vecAddKernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

// Exported C API. Returns 0 on success, non-zero on failure.
extern "C" VECADD_API int vecAdd_host(const float* a_host, const float* b_host, float* out_host, int n) {
    if (!a_host || !b_host || !out_host || n <= 0) return 1;

    const size_t bytes = size_t(n) * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

    cudaError_t err = cudaMalloc((void**)&d_a, bytes);
    if (err != cudaSuccess) { checkCuda(err, "cudaMalloc d_a"); return 2; }
    err = cudaMalloc((void**)&d_b, bytes);
    if (err != cudaSuccess) { checkCuda(err, "cudaMalloc d_b"); cudaFree(d_a); return 3; }
    err = cudaMalloc((void**)&d_out, bytes);
    if (err != cudaSuccess) { checkCuda(err, "cudaMalloc d_out"); cudaFree(d_a); cudaFree(d_b); return 4; }

    err = cudaMemcpy(d_a, a_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { checkCuda(err, "cudaMemcpy a"); goto cleanup; }
    err = cudaMemcpy(d_b, b_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { checkCuda(err, "cudaMemcpy b"); goto cleanup; }

    const int block = 256;
    const int grid = (n + block - 1) / block;
    vecAddKernel<<<grid, block>>>(d_a, d_b, d_out, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) { checkCuda(err, "kernel launch"); goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { checkCuda(err, "cudaDeviceSynchronize"); goto cleanup; }

    err = cudaMemcpy(out_host, d_out, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { checkCuda(err, "cudaMemcpy out"); goto cleanup; }

    // success
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return 0;

cleanup:
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_out) cudaFree(d_out);
    return 5;
}
