// vector_add_test.cu
// Minimal CUDA vector add test: compiles to an executable that
// performs a GPU vector addition and verifies the result on the host.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

__global__ void vecAddKernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 16; // 65536 elements (small, fast)
    const size_t bytes = N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_out) { fprintf(stderr, "Host alloc failed\n"); return 2; }

    for (int i = 0; i < N; ++i) { h_a[i] = (float)i; h_b[i] = (float)(2*i); }

    float *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    checkCuda(cudaMalloc((void**)&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc((void**)&d_b, bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void**)&d_out, bytes), "cudaMalloc d_out");

    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h->d a");
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h->d b");

    const int block = 256;
    const int grid = (N + block - 1) / block;
    vecAddKernel<<<grid, block>>>(d_a, d_b, d_out, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d->h out");

    // verify
    for (int i = 0; i < N; ++i) {
        float expect = h_a[i] + h_b[i];
        if (h_out[i] != expect) {
            fprintf(stderr, "Mismatch at %d: got %f expected %f\n", i, h_out[i], expect);
            return 3;
        }
    }

    printf("vector_add_test: SUCCESS (%d elements)\n", N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}
