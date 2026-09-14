#pragma once
// MFMA bf16 GEMM tile helpers for gfx942 (CDNA3 / MI300X).
//
// Computes C = A @ W^T, i.e. an nn.Linear projection: A is [M, in] activations,
// W is [out, in] weights (BOTH row-major, bf16). Because W is [out,in], the GEMM's
// B-matrix B[k][n] = W[n][k], so A and W are read with the SAME index pattern —
// the projection maps cleanly onto the 16x16x16 MFMA tile.
//
// 16x16x16 layout (confirmed bit-exact against a reference GEMM):
//   lane l (0..63): rc = l%16, grp = l/16 (0..3)
//   inputs : element [row=rc][k = k0 + grp*4 + i], i in 0..3   (row = m for A, n for W)
//   output : acc[i] = C[row = m0 + grp*4 + i][col = n0 + rc]
//
// M (batch) must be padded to a multiple of 16; out/in are multiples of 16 for
// Qwen3-0.6B (Q=2048, KV=1024, HID=1024, INT=3072).

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM) || defined(__HIPCC__)

typedef __bf16 mq_bf16x4 __attribute__((ext_vector_type(4)));
typedef float  mq_f32x4  __attribute__((ext_vector_type(4)));

// Accumulate one 16x16 output tile (rows [m0,m0+16), cols [n0,n0+16)) over the
// full K = in_dim. Wavefront-wide (call with 64 threads). `acc` carries 4 f32/lane.
__device__ __forceinline__ void mq_mfma_tile_accum(
    const __bf16* __restrict__ A,   // [M, in] row-major
    const __bf16* __restrict__ W,   // [out, in] row-major
    int in_dim, int m0, int n0, mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * in_dim;
    const __bf16* wrow = W + (n0 + rc) * in_dim;
    for (int k0 = 0; k0 < in_dim; k0 += 16) {
        mq_bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = arow[k0 + grp * 4 + i];
            b[i] = wrow[k0 + grp * 4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
}

// Accumulate a 16x16 output tile but reduce only over K in [k0, k1) (both
// multiples of 16). Used by the XCD-sharded "contract" projections (O-proj,
// down-proj), where each XCD owns a K sub-range and produces a partial that is
// later all-reduced across XCDs. Identical math to mq_mfma_tile_accum with a
// restricted K loop.
__device__ __forceinline__ void mq_mfma_tile_accum_krange(
    const __bf16* __restrict__ A,   // [M, in] row-major
    const __bf16* __restrict__ W,   // [out, in] row-major
    int in_dim, int k0, int k1, int m0, int n0, mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * in_dim;
    const __bf16* wrow = W + (n0 + rc) * in_dim;
    for (int kk = k0; kk < k1; kk += 16) {
        mq_bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = arow[kk + grp * 4 + i];
            b[i] = wrow[kk + grp * 4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
}

// Store a 16x16 tile: acc[i] -> C[m0 + grp*4 + i][n0 + rc]. T = float or __bf16.
template <typename T>
__device__ __forceinline__ void mq_mfma_tile_store(
    T* __restrict__ C, int ld_out, int m0, int n0, const mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int cn = lane & 15, grp = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp * 4 + i) * ld_out + (n0 + cn)] = (T)acc[i];
}

// Store with a fused residual add: C[r][c] = acc + resid[r][c]. `resid` is an
// f32 [M, out] buffer with the same leading dim (ld_out) as C. Used for the
// O-proj and down-proj epilogues (out = acc + pre-op residual). T = float/__bf16.
template <typename T>
__device__ __forceinline__ void mq_mfma_tile_store_add(
    T* __restrict__ C, const float* __restrict__ resid, int ld_out,
    int m0, int n0, const mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int cn = lane & 15, grp = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = m0 + grp * 4 + i, c = n0 + cn;
        C[r * ld_out + c] = (T)(acc[i] + resid[r * ld_out + c]);
    }
}

// =============================================================================
// fp8 WEIGHT-ONLY path (W8A16): activations stay bf16, weights are e4m3fnuz.
// Cuts weight HBM traffic ~2x on the bandwidth-bound decode. The weight is
// dequantized per-output-channel: W_bf16[n][k] = W_fp8[n][k] * s[n]. Since s[n]
// factors out of the K-sum, the MFMA runs on the *unscaled* fp8->bf16 weight and
// s[n] is folded into the store epilogue (one scale/lane/tile). MI300 fp8 is
// FNUZ (max normal ~240); torch dtype float8_e4m3fnuz is bit-compatible with
// __hip_fp8_e4m3_fnuz. Accuracy validated (probe_fp8_accuracy.py: 40/40 identical
// greedy vs bf16). See project memory "#28 fp8".
#include <hip/hip_fp8.h>
typedef __hip_fp8_e4m3_fnuz mq_fp8;

// Accumulate one 16x16 tile over full K with bf16 A and fp8 W (converted to bf16
// per element before the bf16 MFMA). No scale applied here — folded into store.
__device__ __forceinline__ void mq_mfma_tile_accum_fp8(
    const __bf16* __restrict__ A,   // [M, in] row-major bf16
    const mq_fp8* __restrict__ W,   // [out, in] row-major fp8 e4m3fnuz
    int in_dim, int m0, int n0, mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * in_dim;
    const mq_fp8* wrow = W + (n0 + rc) * in_dim;
    for (int k0 = 0; k0 < in_dim; k0 += 16) {
        mq_bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = arow[k0 + grp * 4 + i];
            b[i] = (__bf16)(float)wrow[k0 + grp * 4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
}

// fp8 K-range variant (for XCD-sharded contract projections: O-proj, down-proj).
__device__ __forceinline__ void mq_mfma_tile_accum_krange_fp8(
    const __bf16* __restrict__ A,   // [M, in] row-major bf16
    const mq_fp8* __restrict__ W,   // [out, in] row-major fp8 e4m3fnuz
    int in_dim, int k0, int k1, int m0, int n0, mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * in_dim;
    const mq_fp8* wrow = W + (n0 + rc) * in_dim;
    for (int kk = k0; kk < k1; kk += 16) {
        mq_bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = arow[kk + grp * 4 + i];
            b[i] = (__bf16)(float)wrow[kk + grp * 4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
}

// Store a 16x16 tile with a per-output-channel dequant scale folded in:
// C[m0+grp*4+i][n0+rc] = acc[i] * scale[n0+rc]. Each lane owns one output column
// (n0+rc) across its 4 rows, so it reads a single scale value. `scale` is the
// full per-output-channel vector (length out_dim). T = float or __bf16.
template <typename T>
__device__ __forceinline__ void mq_mfma_tile_store_scaled(
    T* __restrict__ C, const float* __restrict__ scale, int ld_out,
    int m0, int n0, const mq_f32x4& acc) {
    int lane = threadIdx.x & 63;
    int cn = lane & 15, grp = lane >> 4;
    float s = scale[n0 + cn];
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp * 4 + i) * ld_out + (n0 + cn)] = (T)(acc[i] * s);
}

#endif  // ROCm
