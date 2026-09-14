#pragma once
// =============================================================================
// port.cuh — CUDA (NVIDIA) <-> HIP/ROCm (AMD CDNA3, gfx942) portability shim
// -----------------------------------------------------------------------------
// The MegaQwen megakernel was written for NVIDIA (RTX 3090, sm_86). This header
// isolates every construct that does NOT translate automatically under torch's
// cpp_extension auto-hipify:
//   * wavefront size (NVIDIA warp = 32, AMD CDNA wavefront = 64)
//   * cp.async / pipeline async-copy (no clean CDNA3 equivalent)
//   * cooperative-launch grid sizing (must fill the actual device, not a
//     hardcoded SM count)
// Everything else (headers, __nv_bfloat16, cooperative_groups,
// cudaLaunchCooperativeKernel, __shfl_*_sync, __ldg) is left as-is and relies on
// hipify's textual rewrite on the ROCm PyTorch build.
// =============================================================================

// --- platform detection ------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM) || defined(__HIPCC__)
  #define MQ_ROCM 1
#else
  #define MQ_ROCM 0
#endif

// NOTE: torch's load_inline hipify rewrites the inline .cu SOURCE STRING but NOT
// the local headers that string #includes (config.cuh and this file). So every
// disk header must be platform-explicit about its includes AND provide the CUDA
// type names it uses — we cannot rely on hipify for anything included via -I.
#if MQ_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_cooperative_groups.h>
  // HIP has no __nv_bfloat16; alias it so disk headers/kernels that name the CUDA
  // type resolve. (The hipified source string may already read __hip_bfloat16 —
  // both then resolve to the same type.)
  using __nv_bfloat16 = __hip_bfloat16;
#else
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <cuda_pipeline.h>   // cp.async intrinsics; NVIDIA-only
  #include <cooperative_groups.h>
#endif

#include <cstdlib>   // getenv, atoi (host: grid-size override)
#include <cstdio>    // fprintf (host: grid diagnostic)

// -----------------------------------------------------------------------------
// Wavefront / warp size (COMPILE-TIME).
//
// Deliberately named WARP_SIZE so the (large) existing body of code that already
// references WARP_SIZE symbolically — reduction trees, NUM_WARPS = BLOCK_SIZE /
// WARP_SIZE, per-lane register arrays sized HEAD_DIM / WARP_SIZE, attention
// lane-pairing (idx / WARP_SIZE, idx % WARP_SIZE) — recompiles correctly for a
// 64-wide wavefront with no call-site churn. config.cuh must NOT redefine it.
//
// HIP's builtin `warpSize` is a non-constexpr device value and cannot size
// register arrays, so a real compile-time constant is required.
// -----------------------------------------------------------------------------
#if MQ_ROCM
constexpr int WARP_SIZE = 64;
// HIP's warp-sync builtins (HIP_ENABLE_WARP_SYNC_BUILTINS) static_assert that the
// mask is a 64-bit integer for the 64-wide wavefront — a 32-bit 0xffffffff fails
// to compile. Use WARP_FULL_MASK at every __shfl_*_sync / vote-mask site.
#define WARP_FULL_MASK 0xffffffffffffffffULL
#else
constexpr int WARP_SIZE = 32;
#define WARP_FULL_MASK 0xffffffffu
#endif

// Read-only global load. CUDA's __ldg has overloads for many builtin/vector
// types (uint4, __nv_bfloat16, ...); HIP's does not, and CDNA has no read-only
// texture path anyway. LDG() is a template that emits __ldg on CUDA and a plain
// (compiler-vectorized) load on ROCm. Call sites use LDG(ptr) instead of __ldg.
template <typename T>
__device__ __forceinline__ T LDG(const T* ptr) {
#if MQ_ROCM
    return *ptr;
#else
    return __ldg(ptr);
#endif
}

// -----------------------------------------------------------------------------
// Async global->shared copy shim (replaces cp.async / __pipeline_memcpy_async).
//
// NVIDIA: real asynchronous copy via the pipeline intrinsics.
// AMD (v1, correctness-first): a SYNCHRONOUS 16-byte vectorized copy; commit /
//   wait become no-ops. Semantically identical, just not overlapped. A true
//   async path via __builtin_amdgcn_global_load_lds is a perf follow-up (see
//   DEVLOG); gate it behind MQ_ROCM_ASYNC_LDS once validated.
//
// Call sites should use these mq_* names instead of __pipeline_* / cp.async PTX.
// Each transfer moves 16 bytes (128 bits), matching the kernel's tile chunking.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mq_async_copy16(void* smem_dst, const void* gmem_src) {
#if MQ_ROCM
    // Synchronous 128-bit copy. Correct on CDNA3; overlap added later.
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, 16);
#endif
}

__device__ __forceinline__ void mq_async_commit() {
#if MQ_ROCM
    // no-op: the AMD copy above already completed
#else
    __pipeline_commit();
#endif
}

__device__ __forceinline__ void mq_async_wait_prior(int n) {
#if MQ_ROCM
    (void)n;   // no-op: synchronous copy
#else
    // __pipeline_wait_prior takes a compile-time constant; expand the small set
    // the kernel actually uses (0, 1, 2).
    if (n <= 0)      __pipeline_wait_prior(0);
    else if (n == 1) __pipeline_wait_prior(1);
    else             __pipeline_wait_prior(2);
#endif
}

__device__ __forceinline__ void mq_async_wait_all() {
#if MQ_ROCM
    // no-op
#else
    __pipeline_wait_prior(0);
#endif
}

// -----------------------------------------------------------------------------
// Cooperative-launch grid sizing.
//
// grid.sync() requires every block to be co-resident. The original code
// hardcoded 82 blocks (RTX 3090 SM count); MI300X has 304 CUs. Query the device
// and multiply CU/SM count by the kernel's max active blocks per CU so the
// cooperative grid fills the GPU without over-subscribing (which would deadlock
// grid.sync). Host-side helper; call from the launch wrapper.
// -----------------------------------------------------------------------------
__host__ inline int mq_coop_grid_blocks(const void* kernel, int block_size, size_t dyn_smem_bytes) {
    // Tuning override: MQ_GRID_BLOCKS forces the cooperative grid size. The
    // grid.sync() barrier cost scales with block count, and for a tiny batch-1
    // model, filling the device (CUs x occupancy) can make the barrier dominate.
    // Lets us sweep grid size without recompiling (grid is a launch-time param).
    if (const char* env = getenv("MQ_GRID_BLOCKS")) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    int dev = 0, num_units = 0, blocks_per_unit = 0;
#if MQ_ROCM
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&num_units, hipDeviceAttributeMultiprocessorCount, dev);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_unit, kernel, block_size, dyn_smem_bytes);
#else
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_units, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_unit, kernel, block_size, dyn_smem_bytes);
#endif
    if (blocks_per_unit < 1) blocks_per_unit = 1;
    int blocks = num_units * blocks_per_unit;
    int chosen = blocks > 0 ? blocks : (num_units > 0 ? num_units : 1);
    // These flat (non-sharded) batch-1 decode megakernels are grid.sync-barrier-
    // bound: every block joins one cross-XCD barrier ~140x/token, and that barrier
    // gets more expensive as the grid grows, so filling the device is a net loss
    // (throughput collapses well before device-fill). Cap to a modest grid near
    // the kernel's original design point (~82 = 3090 SM count). Override with
    // MQ_GRID_BLOCKS for other models/GPUs; XCD-sharded kernels use the _fill
    // variant below, which does want to fill the device.
    const int MQ_DEFAULT_COOP_BLOCKS = 76;
    if (chosen > MQ_DEFAULT_COOP_BLOCKS) chosen = MQ_DEFAULT_COOP_BLOCKS;
    return chosen;
}

// Cooperative grid sizing for XCD-sharded kernels. Unlike the flat kernels above,
// the sharded megakernel keeps most barriers intra-XCD and each block's work stays
// L2-local to its XCD, so a device-filling grid is a WIN (more MFMA tiles in flight
// without the flat-barrier penalty). Default to one co-resident block per CU:
// occupancy>=1 guarantees co-residency, so grid.sync stays deadlock-free. Round
// DOWN to a multiple of `mult` (block->XCD = blockIdx.x % mult) so every XCD gets
// an equal block count. MQ_GRID_BLOCKS still overrides (e.g. to probe >1 block/CU).
__host__ inline int mq_coop_grid_blocks_fill(const void* kernel, int block_size,
                                             size_t dyn_smem_bytes, int mult) {
    if (const char* env = getenv("MQ_GRID_BLOCKS")) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    int dev = 0, num_units = 0, blocks_per_unit = 0;
#if MQ_ROCM
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&num_units, hipDeviceAttributeMultiprocessorCount, dev);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_unit, kernel, block_size, dyn_smem_bytes);
#else
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_units, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_unit, kernel, block_size, dyn_smem_bytes);
#endif
    if (mult < 1) mult = 1;
    if (getenv("MQ_PRINT_OCC"))
        fprintf(stderr, "[MQ_OCC] num_units(CU)=%d blocks_per_unit=%d block_size=%d\n",
                num_units, blocks_per_unit, block_size);
    // If the kernel can't fit even one block per CU, fall back to the capped grid
    // (device-fill would exceed residency and deadlock grid.sync).
    if (blocks_per_unit < 1 || num_units < 1)
        return mq_coop_grid_blocks(kernel, block_size, dyn_smem_bytes);
    // Occupancy lever: co-reside k blocks per CU to raise wavefronts/CU (more MLP to
    // hide HBM latency at low batch). Capped at blocks_per_unit so all blocks stay
    // co-resident -> grid.sync remains deadlock-free. Default k=1 (prior behavior).
    int per_cu = 1;
    if (const char* ke = getenv("MQ_BLOCKS_PER_CU")) {
        int k = atoi(ke);
        if (k > 1) per_cu = (k <= blocks_per_unit) ? k : blocks_per_unit;
    }
    int fill = (num_units * per_cu / mult) * mult;   // per_cu blocks per CU, XCD-balanced
    if (fill < mult) fill = mult;
    return fill;
}
