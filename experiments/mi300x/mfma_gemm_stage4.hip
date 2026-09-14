// mfma_gemm_stage4.hip — Stage-4 inner-loop de-risk spike (gfx942 / MI300X).
//
// The production MFMA path (csrc/megakernel/mfma.cuh :: mq_mfma_tile_accum) reads
// A and W straight from global memory with 4 SCALAR __bf16 loads per 16-wide
// K-step and no LDS reuse (A rows are re-streamed for every N-tile). This spike
// measures, bit-exact against that exact scalar loop, how much three inner-loop
// rewrites buy — before any of them touch the working kernel:
//
//   V0  scalar   : the current mq_mfma_tile_accum, verbatim (reference).
//   V1  vec      : same math, one 64-bit bf16x4 vector load instead of 4 scalars.
//   V2  lds      : block stages the A K-chunk into LDS once, reused by 4 waves
//                  (4 N-tiles per block) -> A HBM traffic /4.
//   V3  lds_db   : V2 + __builtin_amdgcn_global_load_lds double-buffering
//                  (prefetch next A chunk into a 2nd LDS buffer during MFMA).
//
// All variants compute C = A[M,K] @ W[N,K]^T (nn.Linear, W row-major [out,in]),
// f32 accumulate, exactly the projection layout the megakernel uses.
//
// Build (inside a ROCm container, gfx942):
//   export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib:$LD_LIBRARY_PATH
//   hipcc --offload-arch=gfx942 -O3 mfma_gemm_stage4.hip -o /tmp/g4 && /tmp/g4

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float  f32x4  __attribute__((ext_vector_type(4)));

#define CK(x) do{ hipError_t e=(x); if(e){printf("HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e));exit(1);} }while(0)

// Problem: Q-proj-like shape, all mult of 16. 2*M*N*K flops.
#ifndef MROWS
#define MROWS 64
#endif
static const int M = MROWS;  // padded batch (sweep via -DMROWS=...)
static const int K = 2048;   // in_dim
static const int N = 2048;   // out_dim

// ------------------------------------------------------------------ V0 scalar
__global__ void gemm_v0(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    int tile = blockIdx.x;                 // one wave per output tile
    int m0 = (tile / (N/16)) * 16;
    int n0 = (tile % (N/16)) * 16;
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * K;
    const __bf16* wrow = W + (n0 + rc) * K;
    f32x4 acc = {0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) { a[i] = arow[k0 + grp*4 + i]; b[i] = wrow[k0 + grp*4 + i]; }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
    int cn = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp*4 + i) * N + (n0 + cn)] = acc[i];
}

// ------------------------------------------------------------------ V1 vec load
__global__ void gemm_v1(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    int tile = blockIdx.x;
    int m0 = (tile / (N/16)) * 16;
    int n0 = (tile % (N/16)) * 16;
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * K;
    const __bf16* wrow = W + (n0 + rc) * K;
    f32x4 acc = {0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        // (k0 + grp*4) is a multiple of 4 elems -> 8-byte aligned -> one 64-bit load.
        bf16x4 a = *reinterpret_cast<const bf16x4*>(arow + k0 + grp*4);
        bf16x4 b = *reinterpret_cast<const bf16x4*>(wrow + k0 + grp*4);
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
    int cn = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp*4 + i) * N + (n0 + cn)] = acc[i];
}

// ------------------------------------------------------------------ V2 LDS A-reuse
// Block = 256 threads = 4 waves. One m0 (16 rows), 4 consecutive n0 tiles (64 cols).
// A[16 x 16] K-chunk staged into LDS once per k0, reused by all 4 waves.
__global__ void gemm_v2(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    __shared__ __bf16 As[16 * 16];
    int mtile = blockIdx.x / (N/64);
    int ntile = blockIdx.x % (N/64);
    int m0 = mtile * 16;
    int wv = threadIdx.x >> 6;             // 0..3
    int n0 = ntile * 64 + wv * 16;
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* wrow = W + (n0 + rc) * K;
    f32x4 acc = {0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        // 256 threads cooperatively load the 16x16 A chunk: As[r*16 + c].
        int t = threadIdx.x;               // 0..255 == 16*16
        As[t] = A[(m0 + (t >> 4)) * K + k0 + (t & 15)];
        __syncthreads();
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = As[rc * 16 + grp*4 + i];
            b[i] = wrow[k0 + grp*4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        __syncthreads();
    }
    int cn = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp*4 + i) * N + (n0 + cn)] = acc[i];
}

// ------------------------------------------------------------------ V4 K-split ILP
// Single output tile per wave (like V0) but 4 independent accumulators, each
// summing a strided quarter of K -> breaks the dependent MFMA chain (4-way ILP).
// Final sum reorders f32 adds, so NOT bit-exact vs V0 (tolerance parity).
__global__ void gemm_v4(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    int tile = blockIdx.x;
    int m0 = (tile / (N/16)) * 16;
    int n0 = (tile % (N/16)) * 16;
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * K;
    const __bf16* wrow = W + (n0 + rc) * K;
    f32x4 a0={0,0,0,0}, a1={0,0,0,0}, a2={0,0,0,0}, a3={0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 64) {
        bf16x4 av0 = *(const bf16x4*)(arow+k0    +grp*4), bv0 = *(const bf16x4*)(wrow+k0    +grp*4);
        bf16x4 av1 = *(const bf16x4*)(arow+k0+16 +grp*4), bv1 = *(const bf16x4*)(wrow+k0+16 +grp*4);
        bf16x4 av2 = *(const bf16x4*)(arow+k0+32 +grp*4), bv2 = *(const bf16x4*)(wrow+k0+32 +grp*4);
        bf16x4 av3 = *(const bf16x4*)(arow+k0+48 +grp*4), bv3 = *(const bf16x4*)(wrow+k0+48 +grp*4);
        a0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av0, bv0, a0, 0,0,0);
        a1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av1, bv1, a1, 0,0,0);
        a2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av2, bv2, a2, 0,0,0);
        a3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av3, bv3, a3, 0,0,0);
    }
    f32x4 acc = a0 + a1 + a2 + a3;
    int cn = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp*4 + i) * N + (n0 + cn)] = acc[i];
}

// ------------------------------------------------------------------ V5 N-block ILP (NN=4)
// One wave computes 4 consecutive N-tiles from a SINGLE A load per k-step ->
// 4-way ILP (independent chains) + A reused 4x from registers, no LDS. Each
// element's K-chain is in-order, so bit-exact vs V0.
template <int NN>
__global__ void gemm_vN(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    int tile = blockIdx.x;
    int m0 = (tile / (N/(16*NN))) * 16;
    int n0 = (tile % (N/(16*NN))) * (16*NN);
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* arow = A + (m0 + rc) * K;
    const __bf16* wrow[NN];
    #pragma unroll
    for (int j = 0; j < NN; j++) wrow[j] = W + (n0 + j*16 + rc) * K;
    f32x4 acc[NN];
    #pragma unroll
    for (int j = 0; j < NN; j++) acc[j] = f32x4{0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        bf16x4 a = *(const bf16x4*)(arow + k0 + grp*4);
        #pragma unroll
        for (int j = 0; j < NN; j++) {
            bf16x4 b = *(const bf16x4*)(wrow[j] + k0 + grp*4);
            acc[j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[j], 0,0,0);
        }
    }
    int cn = lane & 15;
    #pragma unroll
    for (int j = 0; j < NN; j++)
        #pragma unroll
        for (int i = 0; i < 4; i++)
            C[(m0 + grp*4 + i) * N + (n0 + j*16 + cn)] = acc[j][i];
}

#ifdef WITH_V3
// ------------------------------------------------------------------ V3 LDS + global_load_lds double buffer
// Same tiling as V2, but the A chunk is brought in via __builtin_amdgcn_global_load_lds
// (direct global->LDS, no register round-trip) and the NEXT chunk is prefetched
// during the current MFMA. 256 threads x 4 bytes = 1024 B == 16*16*4? no: 16*16
// bf16 = 512 B. So only the first 128 lanes move 4 B each (128*4=512). Guard by tid.
__global__ void gemm_v3(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                        float* __restrict__ C) {
    __shared__ __bf16 As[2][16 * 16];      // double buffer
    int mtile = blockIdx.x / (N/64);
    int ntile = blockIdx.x % (N/64);
    int m0 = mtile * 16;
    int wv = threadIdx.x >> 6;
    int n0 = ntile * 64 + wv * 16;
    int lane = threadIdx.x & 63, rc = lane & 15, grp = lane >> 4;
    const __bf16* wrow = W + (n0 + rc) * K;
    f32x4 acc = {0,0,0,0};

    // Each dword = 2 bf16. 16x16 bf16 = 256 elems = 128 dwords. Lanes 0..127 move one dword.
    int tid = threadIdx.x;
    auto load_chunk = [&](int buf, int k0){
        if (tid < 128) {
            int r = (tid * 2) >> 4;        // element index e=tid*2 -> row e/16
            int c = (tid * 2) & 15;        //                         col e%16
            const __bf16* src = A + (m0 + r) * K + k0 + c;   // 2 contiguous bf16 = 1 dword
            __bf16* dst = &As[buf][r * 16 + c];
            __builtin_amdgcn_global_load_lds((void*)const_cast<__bf16*>(src), (void*)dst, 4, 0, 0);
        }
    };

    int nchunks = K / 16;
    load_chunk(0, 0);
    __builtin_amdgcn_s_waitcnt(0);         // vmcnt/lgkmcnt drain for the LDS fill
    __syncthreads();
    for (int c = 0; c < nchunks; c++) {
        int cur = c & 1, nxt = (c + 1) & 1;
        int k0 = c * 16;
        if (c + 1 < nchunks) load_chunk(nxt, (c + 1) * 16);   // prefetch during MFMA
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = As[cur][rc * 16 + grp*4 + i];
            b[i] = wrow[k0 + grp*4 + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
    }
    int cn = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) C[(m0 + grp*4 + i) * N + (n0 + cn)] = acc[i];
}
#endif  // WITH_V3

template <typename F>
double time_ms(F launch, int iters) {
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    launch();                              // warmup
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(t0));
    for (int i = 0; i < iters; i++) launch();
    CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
    float ms = 0; CK(hipEventElapsedTime(&ms, t0, t1));
    return ms / iters;
}

int main() {
    size_t szA = (size_t)M*K, szW = (size_t)N*K, szC = (size_t)M*N;
    std::vector<__bf16> hA(szA), hW(szW);
    srand(1234);
    for (auto& x : hA) x = (__bf16)((rand()/(float)RAND_MAX - 0.5f) * 0.2f);
    for (auto& x : hW) x = (__bf16)((rand()/(float)RAND_MAX - 0.5f) * 0.2f);
    __bf16 *dA, *dW; float *dC;
    CK(hipMalloc(&dA, szA*2)); CK(hipMalloc(&dW, szW*2)); CK(hipMalloc(&dC, szC*4));
    CK(hipMemcpy(dA, hA.data(), szA*2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, hW.data(), szW*2, hipMemcpyHostToDevice));

    int tiles = (M/16)*(N/16);
    int blk64 = tiles;                 // V0/V1/V4: one 64-thread wave per tile
    int blk256 = (M/16)*(N/64);        // V2/V3: 256-thread block, 4 tiles
    int blkN4 = (M/16)*(N/64);         // V5: 1 wave, 4 N-tiles
    int blkN8 = (M/16)*(N/128);        // V6: 1 wave, 8 N-tiles

    std::vector<float> ref(szC), got(szC);
    auto run_ref = [&](){ hipLaunchKernelGGL(gemm_v0, dim3(blk64), dim3(64), 0, 0, dA, dW, dC); };
    run_ref(); CK(hipDeviceSynchronize());
    CK(hipMemcpy(ref.data(), dC, szC*4, hipMemcpyDeviceToHost));

    struct V { const char* name; std::function<void()> fn; };
    std::vector<V> vs = {
        {"V0_scalar", [&](){ hipLaunchKernelGGL(gemm_v0, dim3(blk64), dim3(64),  0,0, dA,dW,dC); }},
        {"V1_vec",    [&](){ hipLaunchKernelGGL(gemm_v1, dim3(blk64), dim3(64),  0,0, dA,dW,dC); }},
        {"V4_ksplit", [&](){ hipLaunchKernelGGL(gemm_v4, dim3(blk64), dim3(64),  0,0, dA,dW,dC); }},
        {"V5_nblk4",  [&](){ hipLaunchKernelGGL(gemm_vN<4>, dim3(blkN4),dim3(64),0,0, dA,dW,dC); }},
        {"V6_nblk8",  [&](){ hipLaunchKernelGGL(gemm_vN<8>, dim3(blkN8),dim3(64),0,0, dA,dW,dC); }},
        {"V2_lds",    [&](){ hipLaunchKernelGGL(gemm_v2, dim3(blk256),dim3(256), 0,0, dA,dW,dC); }},
#ifdef WITH_V3
        {"V3_lds_db", [&](){ hipLaunchKernelGGL(gemm_v3, dim3(blk256),dim3(256), 0,0, dA,dW,dC); }},
#endif
    };
    double gflop = 2.0 * M * N * K / 1e9;
    printf("shape M=%d K=%d N=%d  (%.3f GFLOP/call)\n", M, K, N, gflop);
    for (auto& v : vs) {
        CK(hipMemset(dC, 0, szC*4));
        v.fn(); CK(hipDeviceSynchronize());
        CK(hipMemcpy(got.data(), dC, szC*4, hipMemcpyDeviceToHost));
        double maxabs = 0, maxrel = 0;
        for (size_t i = 0; i < szC; i++) {
            double d = fabs(got[i] - ref[i]); maxabs = fmax(maxabs, d);
            double r = d / (fabs(ref[i]) + 1e-6); maxrel = fmax(maxrel, r);
        }
        double ms = time_ms(v.fn, 300);
        // GFLOP / ms = 1e9 flop / 1e-3 s = 1e12 flop/s = TFLOP/s.
        printf("%-10s  ms=%.4f  TFLOP/s=%7.2f  maxabs=%.3e maxrel=%.3e  %s\n",
               v.name, ms, gflop / ms, maxabs, maxrel,
               (maxabs < 1e-2 ? "OK" : "MISMATCH"));
    }
    return 0;
}
