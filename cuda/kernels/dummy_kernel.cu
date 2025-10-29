// dummy_kernel.cu
// Placeholder CUDA kernel source for the stereo matcher project.
// This file intentionally contains no functional kernels — it's a scaffold only.

#include <cuda_runtime.h>
#include "../include/dummy_kernel.h"

// Example placeholder: compilation test stub (do not call)
extern "C" __global__ void dummy_kernel_stub(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx]; // no-op
    }
}
