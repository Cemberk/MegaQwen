// S0c spike: intra-XCD hierarchical barrier vs flat cooperative grid.sync().
//
// MI300X = 8 XCD chiplets over Infinity Fabric. A flat grid.sync() is a
// cross-die barrier (the megakernel's dominant cost). S0b established that
// block->XCD placement is deterministic round-robin: blockIdx.x % 8 == XCC_ID.
// Here we prototype a *sense-reversing per-XCD barrier* (blocks synchronize only
// with peers on the same XCD, no cross-fabric traffic) and time it against a
// full grid.sync() at several grid sizes.
//
// Build (inside a ROCm container, gfx942):
//   hipcc --offload-arch=gfx942 -O3 experiments/mi300x/xcd_barrier.hip -o /tmp/xb
//   LD_LIBRARY_PATH=$(python -c "import _rocm_sdk_devel,os;print(os.path.dirname(_rocm_sdk_devel.__file__))")/lib /tmp/xb
// (or point LD_LIBRARY_PATH at the ROCm lib dir).
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define NXCD 8
#define CK(x) do{ hipError_t s_=(x); if(s_){ printf("HIP ERR %s\n", hipGetErrorString(s_)); exit(1);} }while(0)

__device__ __forceinline__ int xcd_id() {
    int xcc;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
    return xcc & (NXCD - 1);
}

// Sense-reversing barrier across all blocks that share one XCD.
//   arrive[NXCD] : per-XCD arrival counter (zeroed between uses by last arriver)
//   sense[NXCD]  : per-XCD released sense value
//   nb_xcd[NXCD] : #blocks mapped to each XCD (host-computed)
//   my_sense     : per-block persistent toggle (block-local register)
__device__ __forceinline__ void xcd_barrier(int* arrive, int* sense,
                                             const int* nb_xcd, int xcd,
                                             bool& my_sense) {
    my_sense = !my_sense;
    if (threadIdx.x == 0) {
        int old = atomicAdd(&arrive[xcd], 1);
        if (old == nb_xcd[xcd] - 1) {
            arrive[xcd] = 0;             // reset for next round
            __threadfence();
            atomicExch(&sense[xcd], (int)my_sense);   // release peers
        } else {
            while (((volatile int*)sense)[xcd] != (int)my_sense) { /* spin */ }
        }
    }
    __syncthreads();
}

// Kernel A: R iterations of the intra-XCD barrier.
__global__ void bench_xcd(int* arrive, int* sense, const int* nb_xcd, int R,
                          unsigned long long* acc) {
    int xcd = xcd_id();
    bool s = false;
    unsigned long long x = 0;
    for (int r = 0; r < R; r++) {
        xcd_barrier(arrive, sense, nb_xcd, xcd, s);
        x += (unsigned long long)r;      // trivial work to keep loop live
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *acc = x;
}

// Kernel B: R iterations of a flat cooperative grid.sync().
__global__ void bench_grid(int R, unsigned long long* acc) {
    cg::grid_group grid = cg::this_grid();
    unsigned long long x = 0;
    for (int r = 0; r < R; r++) {
        grid.sync();
        x += (unsigned long long)r;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *acc = x;
}

static float time_launch(void* kern, dim3 g, dim3 b, void** args) {
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    CK(hipLaunchCooperativeKernel(kern, g, b, args, 0, 0));  // warmup
    CK(hipDeviceSynchronize());
    hipEventRecord(s);
    CK(hipLaunchCooperativeKernel(kern, g, b, args, 0, 0));
    hipEventRecord(e); CK(hipEventSynchronize(e));
    float ms = 0; hipEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int R = 5000, TPB = 256;
    int grids[] = {8, 64, 76, 152, 304, 608};

    int *d_arrive, *d_sense, *d_nb; unsigned long long *d_acc;
    CK(hipMalloc(&d_arrive, NXCD * sizeof(int)));
    CK(hipMalloc(&d_sense,  NXCD * sizeof(int)));
    CK(hipMalloc(&d_nb,     NXCD * sizeof(int)));
    CK(hipMalloc(&d_acc, sizeof(unsigned long long)));

    printf("R=%d iters, TPB=%d\n", R, TPB);
    printf("%6s %14s %14s %10s\n", "blocks", "xcd-bar us/it", "grid.sync us/it", "speedup");
    printf("------------------------------------------------------\n");

    for (int gi = 0; gi < (int)(sizeof(grids)/sizeof(int)); gi++) {
        int G = grids[gi];
        // blocks-per-XCD under round-robin blockIdx%8==XCC
        std::vector<int> nb(NXCD, 0);
        for (int b = 0; b < G; b++) nb[b % NXCD]++;
        CK(hipMemcpy(d_nb, nb.data(), NXCD * sizeof(int), hipMemcpyHostToDevice));
        CK(hipMemset(d_arrive, 0, NXCD * sizeof(int)));
        CK(hipMemset(d_sense,  0, NXCD * sizeof(int)));

        int Rv = R;
        void* argsA[] = {&d_arrive, &d_sense, &d_nb, &Rv, &d_acc};
        void* argsB[] = {&Rv, &d_acc};

        float msA = time_launch((void*)bench_xcd,  dim3(G), dim3(TPB), argsA);
        float msB = time_launch((void*)bench_grid, dim3(G), dim3(TPB), argsB);
        double usA = msA * 1e3 / R, usB = msB * 1e3 / R;
        printf("%6d %14.3f %14.3f %9.2fx\n", G, usA, usB, usB / usA);
    }
    hipFree(d_arrive); hipFree(d_sense); hipFree(d_nb); hipFree(d_acc);
    return 0;
}
