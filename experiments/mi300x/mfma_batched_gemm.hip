// S1 building block test: batched projection GEMM C[M,out] = A[M,in] @ W[out,in]^T
// using the reusable tile in csrc/megakernel/mfma.cuh. Verifies correctness on a
// projection shape and measures TFLOP/s on a large GEMM.
// Build (on node): hipcc --offload-arch=gfx942 -O3 -Icsrc/megakernel \
//   experiments/mi300x/mfma_batched_gemm.hip -o /tmp/bgemm && /tmp/bgemm
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mfma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// One wavefront (block of 64) per 16x16 output tile.
__global__ void bgemm(const __bf16* __restrict__ A, const __bf16* __restrict__ W,
                      float* __restrict__ C, int in_dim, int out_dim) {
    int ntiles = out_dim / 16;
    int mt = blockIdx.x / ntiles, nt = blockIdx.x % ntiles;
    mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    mq_mfma_tile_accum(A, W, in_dim, mt * 16, nt * 16, acc);
    mq_mfma_tile_store(C, out_dim, mt * 16, nt * 16, acc);
}

#define CK(x) do{ hipError_t s_=(x); if(s_){ printf("HIP ERROR %s\n", hipGetErrorString(s_)); exit(1);} }while(0)

static void run_correctness(int M, int in_dim, int out_dim) {
    std::vector<__bf16> A(M * in_dim), W(out_dim * in_dim);
    std::vector<float> ref(M * out_dim, 0.f), C(M * out_dim);
    srand(7);
    for (auto& x : A) x = (__bf16)(rand() / (float)RAND_MAX - 0.5f);
    for (auto& x : W) x = (__bf16)(rand() / (float)RAND_MAX - 0.5f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < out_dim; n++) {
            float s = 0.f;
            for (int k = 0; k < in_dim; k++) s += (float)A[m * in_dim + k] * (float)W[n * in_dim + k];
            ref[m * out_dim + n] = s;
        }
    __bf16 *dA, *dW; float* dC;
    CK(hipMalloc(&dA, A.size() * 2)); CK(hipMalloc(&dW, W.size() * 2)); CK(hipMalloc(&dC, C.size() * 4));
    CK(hipMemcpy(dA, A.data(), A.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice));
    bgemm<<<(M / 16) * (out_dim / 16), 64>>>(dA, dW, dC, in_dim, out_dim);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));
    float maxrel = 0.f;
    for (size_t i = 0; i < C.size(); i++)
        maxrel = fmaxf(maxrel, fabsf(C[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f));
    printf("CORRECTNESS M=%d in=%d out=%d  max_rel_err=%.5f  %s\n",
           M, in_dim, out_dim, maxrel, (maxrel < 0.02f ? "PASS" : "FAIL"));
    hipFree(dA); hipFree(dW); hipFree(dC);
}

static void run_perf(int M, int in_dim, int out_dim, int R) {
    __bf16 *dA, *dW; float* dC;
    CK(hipMalloc(&dA, (size_t)M * in_dim * 2));
    CK(hipMalloc(&dW, (size_t)out_dim * in_dim * 2));
    CK(hipMalloc(&dC, (size_t)M * out_dim * 4));
    CK(hipMemset(dA, 0, (size_t)M * in_dim * 2));
    CK(hipMemset(dW, 0, (size_t)out_dim * in_dim * 2));
    int blocks = (M / 16) * (out_dim / 16);
    bgemm<<<blocks, 64>>>(dA, dW, dC, in_dim, out_dim);  // warmup
    CK(hipDeviceSynchronize());
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for (int r = 0; r < R; r++) bgemm<<<blocks, 64>>>(dA, dW, dC, in_dim, out_dim);
    hipEventRecord(e); CK(hipEventSynchronize(e));
    float ms = 0.f; hipEventElapsedTime(&ms, s, e);
    double flops = 2.0 * M * in_dim * (double)out_dim * R;
    printf("PERF M=%d in=%d out=%d R=%d blocks=%d  %.2f ms  %.1f TFLOP/s\n",
           M, in_dim, out_dim, R, blocks, ms, flops / (ms * 1e-3) / 1e12);
    hipFree(dA); hipFree(dW); hipFree(dC);
}

int main() {
    run_correctness(16, 1024, 3072);   // gate/up-like
    run_correctness(32, 1024, 2048);   // Q-like, M=2 tiles
    run_perf(64, 4096, 8192, 100);     // large GEMM throughput
    return 0;
}
