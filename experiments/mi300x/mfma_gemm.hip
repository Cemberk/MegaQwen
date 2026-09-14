// S0a spike: tiled MFMA bf16 GEMM on gfx942 (MI300X).
// Verifies __builtin_amdgcn_mfma_f32_16x16x16bf16_1k semantics vs a reference
// GEMM and measures bf16 matrix-core TFLOP/s. Build (in a ROCm container, gfx942):
//   hipcc --offload-arch=gfx942 -O3 experiments/mi300x/mfma_gemm.hip -o /tmp/mfma_gemm
//   /tmp/mfma_gemm 64
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float  f32x4  __attribute__((ext_vector_type(4)));

// One wavefront (64 lanes) computes D[16x16] = A[16xK] @ B[Kx16], K % 16 == 0.
// CDNA 16x16x16 layout: lane l -> (l%16) is A-row / B-col / D-col; (l/16) is the
// K-block (inputs) / M-block (output), 4 elems per lane.
__global__ void mfma_gemm16(const __bf16* __restrict__ A, const __bf16* __restrict__ B,
                            float* __restrict__ D, int K) {
    int lane = threadIdx.x;
    int rc = lane % 16, grp = lane / 16;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 16) {
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a[i] = A[rc * K + (k0 + grp * 4 + i)];   // A[row=rc][k]
            b[i] = B[(k0 + grp * 4 + i) * 16 + rc];  // B[k][col=rc]
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++)
        D[(grp * 4 + i) * 16 + rc] = acc[i];         // D[row=grp*4+i][col=rc]
}

// Throughput: many wavefronts each run a big-K MFMA loop `iters` times.
__global__ void mfma_perf(const __bf16* __restrict__ A, const __bf16* __restrict__ B,
                          float* __restrict__ out, int K, int iters) {
    int lane = threadIdx.x;
    int rc = lane % 16, grp = lane / 16;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    for (int it = 0; it < iters; it++)
        for (int k0 = 0; k0 < K; k0 += 16) {
            bf16x4 a, b;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = A[rc * K + (k0 + grp * 4 + i)];
                b[i] = B[(k0 + grp * 4 + i) * 16 + rc];
            }
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }
    out[blockIdx.x * 64 + lane] = acc[0];  // sink, prevents dead-code elimination
}

#define CK(x) do{ hipError_t hip_status_=(x); if(hip_status_){ printf("HIP ERROR: %s\n", hipGetErrorString(hip_status_)); exit(1);} }while(0)

int main(int argc, char** argv) {
    int K = argc > 1 ? atoi(argv[1]) : 64;   // multiple of 16
    const int M = 16, N = 16;

    std::vector<float> hA(M * K), hB(K * N), ref(M * N, 0.f);
    std::vector<__bf16> bA(M * K), bB(K * N);
    srand(1234);
    for (int i = 0; i < M * K; i++) { hA[i] = rand() / (float)RAND_MAX - 0.5f; bA[i] = (__bf16)hA[i]; }
    for (int i = 0; i < K * N; i++) { hB[i] = rand() / (float)RAND_MAX - 0.5f; bB[i] = (__bf16)hB[i]; }
    // reference on the bf16-rounded values (device also rounds to bf16)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += (float)bA[m * K + k] * (float)bB[k * N + n];
            ref[m * N + n] = s;
        }

    __bf16 *dA, *dB; float *dD;
    CK(hipMalloc(&dA, M * K * sizeof(__bf16)));
    CK(hipMalloc(&dB, K * N * sizeof(__bf16)));
    CK(hipMalloc(&dD, M * N * sizeof(float)));
    CK(hipMemcpy(dA, bA.data(), M * K * sizeof(__bf16), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, bB.data(), K * N * sizeof(__bf16), hipMemcpyHostToDevice));
    mfma_gemm16<<<1, 64>>>(dA, dB, dD, K);
    CK(hipDeviceSynchronize());
    std::vector<float> hD(M * N);
    CK(hipMemcpy(hD.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost));

    float maxabs = 0.f, maxrel = 0.f;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(hD[i] - ref[i]);
        maxabs = fmaxf(maxabs, e);
        maxrel = fmaxf(maxrel, e / (fabsf(ref[i]) + 1e-6f));
    }
    printf("CORRECTNESS K=%d  max_abs_err=%.5f  max_rel_err=%.5f  D[0]=%.4f ref[0]=%.4f  %s\n",
           K, maxabs, maxrel, hD[0], ref[0], (maxrel < 0.02f ? "PASS" : "FAIL"));

    // perf
    int Kp = 4096, iters = 2000, blocks = 304 * 8;
    __bf16 *pA, *pB; float *pOut;
    CK(hipMalloc(&pA, M * Kp * sizeof(__bf16)));
    CK(hipMalloc(&pB, Kp * N * sizeof(__bf16)));
    CK(hipMalloc(&pOut, (size_t)blocks * 64 * sizeof(float)));
    CK(hipMemset(pA, 0, M * Kp * sizeof(__bf16)));
    CK(hipMemset(pB, 0, Kp * N * sizeof(__bf16)));
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    mfma_perf<<<blocks, 64>>>(pA, pB, pOut, Kp, 20);   // warmup
    CK(hipDeviceSynchronize());
    hipEventRecord(s);
    mfma_perf<<<blocks, 64>>>(pA, pB, pOut, Kp, iters);
    hipEventRecord(e);
    CK(hipEventSynchronize(e));
    float ms = 0.f; hipEventElapsedTime(&ms, s, e);
    double flops = 2.0 * 16 * 16 * (double)Kp * iters * blocks;
    printf("PERF blocks=%d K=%d iters=%d  %.2f ms  %.1f TFLOP/s (bf16 MFMA)\n",
           blocks, Kp, iters, ms, flops / (ms * 1e-3) / 1e12);
    return 0;
}
