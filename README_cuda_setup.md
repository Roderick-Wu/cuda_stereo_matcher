# CUDA kernel development setup (scaffold)

This file explains the minimal steps to start developing and building CUDA kernels for this repository.

Prerequisites (Windows):
- NVIDIA CUDA Toolkit installed and nvcc available on PATH
- CMake (>= 3.18)
- Visual Studio with C++ build tools (matching CUDA-supported MSVC) for MSVC generator

Quick start (from repository root, PowerShell):

1. Configure and build (example):
   .\cuda\build\build_cuda.ps1

2. The build will produce a static library (if tools are present) under `cuda/build/cmake-build`.

Notes:
- This repo currently contains placeholders only. Implement kernels in `cuda/kernels/` and headers in `cuda/include/`.
- If you prefer PyTorch JIT/C++ extensions, consider adding a `setup.py` using `torch.utils.cpp_extension` instead of CMake.
