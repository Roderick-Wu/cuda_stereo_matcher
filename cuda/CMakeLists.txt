cmake_minimum_required(VERSION 3.18)
project(stereo_cuda_kernels LANGUAGES C CXX CUDA)

# Minimal CMake scaffold to compile CUDA kernels into a static library.
# This file is a starting point. Adjust sources and targets for your needs.

file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")

# Exclude test sources from the library so the test's symbols are not defined
# twice (once in the library and once in the test executable). This avoids
# "multiple definition" / nvlink errors on Windows when linking static libs
# that contain CUDA device code.
list(REMOVE_ITEM CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vector_add_test.cu")

# Also exclude the library wrapper source from the generic static library so the
# symbol is only compiled into the shared library target (avoids dllimport/definition
# conflicts when the same symbol is compiled into a static lib and a DLL).
list(REMOVE_ITEM CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vecadd_lib.cu")

add_library(stereo_cuda_kernels STATIC ${CUDA_SOURCES})
target_include_directories(stereo_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(stereo_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Configured stereo_cuda_kernels with sources: ${CUDA_SOURCES}")

# Add a small test executable that builds and runs a vector-add kernel. Useful
# to verify that CUDA/CMake/MSVC toolchain is configured correctly on Windows.
add_executable(vector_add_test "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vector_add_test.cu")
set_target_properties(vector_add_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(vector_add_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vector_add_test PRIVATE stereo_cuda_kernels)


# Build a small shared library that exposes a C API for vector-add so Python (or
# other languages) can call it via ctypes or a C extension. This library contains
# the device kernel and the host wrapper `vecAdd_host`.
add_library(vecadd_shared SHARED "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vecadd_lib.cu")
target_include_directories(vecadd_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(vecadd_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
if (WIN32)
    # Ensure the DLL exports symbols properly on Windows
    target_compile_definitions(vecadd_shared PRIVATE VECADD_EXPORTS)
endif()


