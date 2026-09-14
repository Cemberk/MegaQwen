"""Shared compiler-flag helper for JIT-built kernels (CUDA and HIP/ROCm).

The kernels were written for NVIDIA nvcc. Several flags used across the loaders
are nvcc-only and make hipcc fail on a ROCm PyTorch build:
  --use_fast_math, --expt-relaxed-constexpr, -lineinfo, -maxrregcount=N, -arch=sm_XX

`cuda_cflags()` returns the right `extra_cuda_cflags` for whichever backend the
active torch was built against, so every loader can share one code path.
"""

import torch


def is_rocm() -> bool:
    """True when torch was built against ROCm/HIP (so load_inline uses hipcc)."""
    return getattr(torch.version, "hip", None) is not None


def cuda_cflags(include_dirs=(), nvcc_extra=(), amd_arch="gfx942"):
    """Build the extra_cuda_cflags list for load_inline.

    include_dirs : dirs added as -I on BOTH backends (e.g. the kernel dir).
    nvcc_extra   : nvcc-only flags applied on CUDA only and dropped on ROCm
                   (e.g. "-arch=sm_86", "-lineinfo", "-maxrregcount=64").
    amd_arch     : ROCm target arch (MI300X = gfx942).
    """
    flags = ["-O3", "-std=c++17"]
    if is_rocm():
        # hipcc equivalents; -DUSE_ROCM lets port.cuh pick the AMD code paths
        # even outside device compilation.
        flags += ["-ffast-math", f"--offload-arch={amd_arch}", "-DUSE_ROCM"]
    else:
        flags += ["--use_fast_math", "--expt-relaxed-constexpr"]
        flags += list(nvcc_extra)
    flags += [f"-I{d}" for d in include_dirs]
    return flags


# CUDA math/util libraries -> their ROCm equivalents (for extra_ldflags).
_ROCM_LIB_MAP = {
    "cublas": "hipblas",
    "cublasLt": "hipblaslt",
    "cusparse": "hipsparse",
    "cusolver": "hipsolver",
    "curand": "hiprand",
    "cufft": "hipfft",
}


def ld_flags(cuda_libs=()):
    """Return extra_ldflags, mapping CUDA libs to ROCm equivalents on HIP.

    e.g. ld_flags(["cublas"]) -> ["-lcublas"] on CUDA, ["-lhipblas"] on ROCm.
    hipify rewrites the cuBLAS *symbols* in the source; only the link flag needs
    this manual remap.
    """
    out = []
    for lib in cuda_libs:
        name = _ROCM_LIB_MAP.get(lib, lib) if is_rocm() else lib
        out.append(f"-l{name}")
    return out
