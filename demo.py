"""
Demo: load the vecadd shared library and call vecAdd_host using NumPy arrays.

This demo uses ctypes to call the C API exported by the CUDA shared library.
Run after you build the `vecadd_shared` target with CMake.

On Windows the built DLL will usually be at:
  cuda/build/cmake-build/Release/vecadd_shared.dll

Usage:
  python cuda/demo.py
"""
import sys
import ctypes
import platform
from pathlib import Path
import numpy as np


def find_library_path():
    root = Path(__file__).resolve().parent
    # expected build output path
    candidates = []
    build_dir = root / "build" / "cmake-build"
    if platform.system() == "Windows":
        candidates.append(build_dir / "Release" / "vecadd_shared.dll")
        candidates.append(build_dir / "Debug" / "vecadd_shared.dll")
    else:
        candidates.append(build_dir / "libvecadd_shared.so")
        candidates.append(build_dir / "libvecadd_shared.dylib")

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find vecadd_shared library in {build_dir}. Build it first.")


def main():
    lib_path = find_library_path()
    print("Loading library:", lib_path)
    lib = ctypes.CDLL(str(lib_path))

    # signature: int vecAdd_host(const float* a_host, const float* b_host, float* out_host, int n);
    vecadd = lib.vecAdd_host
    vecadd.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    vecadd.restype = ctypes.c_int

    N = 1 << 10
    a = np.arange(N, dtype=np.float32)
    b = (np.arange(N, dtype=np.float32) * 2.0)
    out = np.empty_like(a)

    # ensure contiguous
    assert a.flags['C_CONTIGUOUS'] and b.flags['C_CONTIGUOUS'] and out.flags['C_CONTIGUOUS']

    a_p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_p = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_p = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = vecadd(a_p, b_p, out_p, ctypes.c_int(N))
    if rc != 0:
        print("vecAdd_host failed with code:", rc)
        sys.exit(1)

    # verify
    if not np.allclose(out, a + b):
        print("Result verification failed")
        mism = np.where(out != a + b)[0][:10]
        print("mismatches:", mism)
        sys.exit(2)

    print("vecAdd via shared lib OK (N=", N, ")")


if __name__ == '__main__':
    main()
