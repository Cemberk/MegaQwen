// S0b spike: can a gfx942 kernel read its physical XCD/XCC id?
// Dumps HW_ID1/HW_ID2 per block and histograms candidate XCC id (HW_ID2[3:0]).
// If ~8 buckets of ~equal size appear, XCC id is readable and blocks spread across
// the 8 XCDs. Build: hipcc --offload-arch=gfx942 -O3 experiments/mi300x/xcd_probe.hip -o /tmp/xcd
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void probe(unsigned* h1, unsigned* h2) {
    if (threadIdx.x == 0) {
        unsigned a = 0, b = 0;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(a));
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(b));
        h1[blockIdx.x] = a;
        h2[blockIdx.x] = b;
    }
}

#define CK(x) do{ hipError_t s_=(x); if(s_){ printf("ERR %s\n", hipGetErrorString(s_)); exit(1);} }while(0)

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 304;
    unsigned *d1, *d2;
    CK(hipMalloc(&d1, n * 4)); CK(hipMalloc(&d2, n * 4));
    probe<<<n, 64>>>(d1, d2);
    CK(hipDeviceSynchronize());
    std::vector<unsigned> a(n), b(n);
    CK(hipMemcpy(a.data(), d1, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), d2, n * 4, hipMemcpyDeviceToHost));

    printf("first 8 blocks:\n");
    for (int i = 0; i < 8 && i < n; i++)
        printf("  blk%-3d HW_ID1=0x%08x HW_ID2=0x%08x  xcc?(h2&0xf)=%u\n", i, a[i], b[i], b[i] & 0xf);
    int cnt[16] = {0};
    for (int i = 0; i < n; i++) cnt[b[i] & 0xf]++;
    printf("XCC_ID (=HW_ID2 & 0xf) histogram over %d blocks: ", n);
    for (int i = 0; i < 16; i++) if (cnt[i]) printf("xcc%d=%d ", i, cnt[i]);
    printf("\n");
    return 0;
}
