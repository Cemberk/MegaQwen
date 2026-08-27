/**
 * xcd.cuh — MI300X XCD-aware primitives for the cooperative megakernel (Stage 2).
 *
 * MI300X = 8 XCD chiplets (~38 CUs each, private L2) over Infinity Fabric = a
 * NUMA hierarchy. A flat cooperative grid.sync() is a cross-die barrier and a
 * dominant megakernel cost (measurably more expensive than an intra-XCD
 * barrier). This header provides:
 *
 *   - mq_xcd_id()       physical XCD/XCC id (0..7), read from hwreg. For a
 *                       cooperative launch, block->XCD placement is a
 *                       deterministic round-robin: blockIdx.x % 8 == XCC_ID.
 *   - mq_xcd_bar(...)   sense-reversing barrier across the blocks that share one
 *                       XCD (no cross-fabric traffic).
 *
 * Cross-XCD synchronization stays on cg::grid_group::sync() — used only at the
 * genuine all-reduce points (O-proj + down-proj partial-sum, final state), the
 * same cut points as Phase-2 tensor-parallel.
 */
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#define MQ_NXCD 8

// Physical XCD id (0..7). gfx942: HW_REG_XCC_ID is valid (HW_ID1/2 are not).
__device__ __forceinline__ int mq_xcd_id() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    int xcc;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
    return xcc & (MQ_NXCD - 1);
#else
    return blockIdx.x % MQ_NXCD;  // fallback (unused off-CDNA)
#endif
}

// Per-XCD sense-reversing barrier state (device global buffers, one slot/XCD).
//   arrive[MQ_NXCD] : arrival counter, reset by the last arriver each round
//   sense[MQ_NXCD]  : released sense value
//   nb[MQ_NXCD]     : #blocks mapped to each XCD (host-computed from grid size)
struct MqXcdBar {
    int* arrive;
    int* sense;
    const int* nb;
};

// Barrier across all blocks sharing XCD `xcd`. `my_sense` is a per-block
// persistent toggle (a register carried across calls). Synchronizes only
// same-XCD peers — no Infinity-Fabric traffic.
__device__ __forceinline__ void mq_xcd_bar(const MqXcdBar& b, int xcd, bool& my_sense) {
    my_sense = !my_sense;
    if (threadIdx.x == 0) {
        int old = atomicAdd(&b.arrive[xcd], 1);
        if (old == b.nb[xcd] - 1) {
            b.arrive[xcd] = 0;
            __threadfence();
            atomicExch(&b.sense[xcd], (int)my_sense);
        } else {
            while (((volatile int*)b.sense)[xcd] != (int)my_sense) { /* spin */ }
        }
    }
    __syncthreads();
}
