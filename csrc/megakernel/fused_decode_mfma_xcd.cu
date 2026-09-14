/**
 * Stage 2 — software XCD-aware batched MFMA decode (MI300X / gfx942).
 *
 * fused_decode_mfma.cu (Stage 1) distributes every phase over ALL grid warps and
 * separates phases with a flat cooperative grid.sync(). On MI300X (8 XCD chiplets
 * over Infinity Fabric) that flat barrier is a cross-die sync and a dominant
 * decode cost. This variant maps the transformer onto the 8 XCDs like a
 * within-kernel tensor-parallel layout, so almost every phase boundary becomes an
 * INTRA-XCD barrier (cheaper than a cross-die sync) and a full cross-XCD
 * grid.sync() survives only at the two genuine all-reduces (O-proj and down-proj
 * partial sums).
 *
 * Sharding (X = 8 XCDs; the GQA grouping lines up exactly):
 *   - XCD x owns Q-heads {2x, 2x+1}, KV-head {x}          (16 Q / 8 KV / 8 XCD)
 *   - XCD x owns intermediate columns [x*384, (x+1)*384)  (3072 / 8)
 *   - The residual stream (hidden, HIDDEN=1024) is REPLICATED per XCD: buffers
 *     hidden/g_normalized/g_residual/g_activations are [X, Mpad, HID]; each XCD
 *     reads & writes only its own replica, so norm/elementwise stay intra-XCD.
 *   - Expand projections (QKV, gate/up) are output-sharded: XCD x writes its own
 *     column slice of the single shared g_q/g_k/g_v/g_gate/g_up buffers.
 *   - Contract projections (O-proj, down-proj) are K-sharded: XCD x reduces only
 *     its own K sub-range into a per-XCD partial [X, Mpad, HID], then a cross-XCD
 *     barrier publishes all partials and every XCD sums them (+residual) into its
 *     replica. These two are the ONLY cross-XCD barriers per layer.
 *
 * Barriers per layer: 2 cross-XCD grid.sync() + 9 intra-XCD (vs 9 flat in Stage 1).
 *
 * MQ_XCD_HIER (default 1): when 0, ALL barriers are grid.sync() — used to validate
 * that the *sharding + all-reduce* is numerically correct independent of the
 * hierarchical barrier. When 1, the 9 local points use the intra-XCD barrier.
 */

#include "config.cuh"
#include "mfma.cuh"
#include "xcd.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef MQ_XCD_HIER
#define MQ_XCD_HIER 1
#endif

constexpr int MFMA_BLOCK_SIZE = 256;
constexpr int MFMA_NUM_WARPS  = MFMA_BLOCK_SIZE / WARP_SIZE;   // 4 on gfx942
constexpr float MFMA_RMS_EPS  = 1e-6f;

constexpr int MFMA_VOCAB_SIZE = 151936;
constexpr int MFMA_LM_BLOCK   = 256;

// Per-XCD shard sizes (compile-time; Qwen3-0.6B divides evenly by 8).
constexpr int XCD_Q_HEADS   = NUM_Q_HEADS  / MQ_NXCD;          // 2
constexpr int XCD_KV_HEADS  = NUM_KV_HEADS / MQ_NXCD;          // 1
constexpr int XCD_Q_COLS    = XCD_Q_HEADS * HEAD_DIM;          // 256 (attn_out / O-proj K slice)
constexpr int XCD_INT       = INTERMEDIATE_SIZE / MQ_NXCD;     // 384

struct MFMALayerWeights {
    // In the fp8 build (MQ_FP8_WEIGHTS=1) the 7 projection slots hold fp8
    // (e4m3fnuz) weight pointers instead of bf16; norms stay bf16. The 7 per-
    // output-channel f32 dequant scales are appended so the struct layout matches
    // the host-side copy exactly for both builds.
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
#if MQ_FP8_WEIGHTS
    const float* q_proj_scale;
    const float* k_proj_scale;
    const float* v_proj_scale;
    const float* o_proj_scale;
    const float* gate_proj_scale;
    const float* up_proj_scale;
    const float* down_proj_scale;
#endif
};

#define BF(p)  reinterpret_cast<const __bf16*>(p)
#define FP8(p) reinterpret_cast<const mq_fp8*>(p)

// =============================================================================
// Small helpers (shared with Stage 1)
// =============================================================================

__device__ __forceinline__ float mfma_to_f(float x) { return x; }
__device__ __forceinline__ float mfma_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void mfma_store_as(float& d, float v) { d = v; }
__device__ __forceinline__ void mfma_store_as(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

__device__ __forceinline__ float mfma_warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o /= 2) v += __shfl_down_sync(WARP_FULL_MASK, v, o);
    return v;
}
__device__ __forceinline__ float mfma_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float mfma_block_reduce_sum(float v, float* red) {
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    v = mfma_warp_reduce_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = (threadIdx.x < MFMA_NUM_WARPS) ? red[threadIdx.x] : 0.0f;
    if (warp == 0) { s = mfma_warp_reduce_sum(s); if (lane == 0) red[0] = s; }
    __syncthreads();
    return red[0];
}

// Intra-XCD local block indexing (blocks with blockIdx.x % 8 == xcd).
__device__ __forceinline__ int mfma_lbid(int xcd) { return blockIdx.x / MQ_NXCD; }
__device__ __forceinline__ int mfma_lnb(int xcd)  { return (gridDim.x - xcd + MQ_NXCD - 1) / MQ_NXCD; }

// =============================================================================
// XCD-sharded GEMM projections
// =============================================================================

// Expand projection (output-sharded, full K): C[:, col0:col0+col_len] over this
// XCD's local warps. A = this XCD's replica [Mpad,in]; W = full [out,in]; C is the
// single shared buffer of width out_dim.
template <typename Tout>
__device__ __forceinline__ void gemm_xcd_out(
    const __bf16* __restrict__ A, const __bf16* __restrict__ W, Tout* __restrict__ C,
    int Mpad, int in_dim, int out_dim, int col0, int col_len, int xcd) {
    int lwarp = mfma_lbid(xcd) * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nlw   = mfma_lnb(xcd) * MFMA_NUM_WARPS;
    int nt    = col_len / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = lwarp; t < ntiles; t += nlw) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum(A, W, in_dim, mi * 16, col0 + ni * 16, acc);
        mq_mfma_tile_store<Tout>(C, out_dim, mi * 16, col0 + ni * 16, acc);
    }
}

// Contract projection (K-sharded): partial[:, :] = A[:, k0:k0+k_len] @ W[:, k0:k0+k_len]^T
// over this XCD's local warps, written to this XCD's partial buffer [Mpad,out_dim].
__device__ __forceinline__ void gemm_xcd_partial(
    const __bf16* __restrict__ A, const __bf16* __restrict__ W, float* __restrict__ Cpart,
    int Mpad, int in_dim, int out_dim, int k0, int k_len, int xcd) {
    int lwarp = mfma_lbid(xcd) * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nlw   = mfma_lnb(xcd) * MFMA_NUM_WARPS;
    int nt    = out_dim / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = lwarp; t < ntiles; t += nlw) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum_krange(A, W, in_dim, k0, k0 + k_len, mi * 16, ni * 16, acc);
        mq_mfma_tile_store<float>(Cpart, out_dim, mi * 16, ni * 16, acc);
    }
}

#if MQ_FP8_WEIGHTS
// fp8 weight-only counterparts of gemm_xcd_out / gemm_xcd_partial: identical
// tiling, but W is fp8 (converted to bf16 in-register) and the per-output-channel
// dequant scale is folded into the store. A (activations) stays bf16.
template <typename Tout>
__device__ __forceinline__ void gemm_xcd_out_fp8(
    const __bf16* __restrict__ A, const mq_fp8* __restrict__ W, const float* __restrict__ scale,
    Tout* __restrict__ C, int Mpad, int in_dim, int out_dim, int col0, int col_len, int xcd) {
    int lwarp = mfma_lbid(xcd) * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nlw   = mfma_lnb(xcd) * MFMA_NUM_WARPS;
    int nt    = col_len / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = lwarp; t < ntiles; t += nlw) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum_fp8(A, W, in_dim, mi * 16, col0 + ni * 16, acc);
        mq_mfma_tile_store_scaled<Tout>(C, scale, out_dim, mi * 16, col0 + ni * 16, acc);
    }
}

__device__ __forceinline__ void gemm_xcd_partial_fp8(
    const __bf16* __restrict__ A, const mq_fp8* __restrict__ W, const float* __restrict__ scale,
    float* __restrict__ Cpart, int Mpad, int in_dim, int out_dim, int k0, int k_len, int xcd) {
    int lwarp = mfma_lbid(xcd) * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nlw   = mfma_lnb(xcd) * MFMA_NUM_WARPS;
    int nt    = out_dim / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = lwarp; t < ntiles; t += nlw) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum_krange_fp8(A, W, in_dim, k0, k0 + k_len, mi * 16, ni * 16, acc);
        // Scale factors out of the K-sum, so applying it to each per-XCD partial is
        // exact: allreduce sums s[n]*partial_xcd = s[n]*full.
        mq_mfma_tile_store_scaled<float>(Cpart, scale, out_dim, mi * 16, ni * 16, acc);
    }
}
#endif  // MQ_FP8_WEIGHTS

// All-reduce the X per-XCD partials (+ residual) into this XCD's output replica.
// partials: [X, Mpad, dim]; resid_rep/out_rep: [X, Mpad, dim]. Local threads.
template <typename Tout>
__device__ __forceinline__ void allreduce_add_resid(
    const float* __restrict__ partials, const float* __restrict__ resid_rep,
    Tout* __restrict__ out_rep, int Mpad, int dim, int xcd) {
    size_t rep = (size_t)Mpad * dim;
    int lt  = mfma_lbid(xcd) * MFMA_BLOCK_SIZE + threadIdx.x;
    int nlt = mfma_lnb(xcd) * MFMA_BLOCK_SIZE;
    const float* resid = resid_rep + (size_t)xcd * rep;
    for (size_t t = lt; t < rep; t += nlt) {
        float s = 0.0f;
        #pragma unroll
        for (int xi = 0; xi < MQ_NXCD; xi++) s += partials[(size_t)xi * rep + t];
        mfma_store_as(out_rep[(size_t)xcd * rep + t], s + resid[t]);
    }
}

// =============================================================================
// Per-replica RMSNorm (this XCD's replica only, all B rows).
// =============================================================================

template <typename Tin, typename Tout>
__device__ void rmsnorm_xcd(
    const Tin* __restrict__ in_rep, const __nv_bfloat16* __restrict__ w,
    Tout* __restrict__ out_rep, float* __restrict__ resid_rep,
    int B, int Mpad, int dim, int xcd) {
    __shared__ float smem[HIDDEN_SIZE];
    __shared__ float red[MFMA_NUM_WARPS];
    size_t rep = (size_t)Mpad * dim;
    const Tin* in = in_rep + (size_t)xcd * rep;
    Tout* out     = out_rep + (size_t)xcd * rep;
    float* resid  = resid_rep ? resid_rep + (size_t)xcd * rep : nullptr;
    int lb = mfma_lbid(xcd), nlb = mfma_lnb(xcd);
    for (int b = lb; b < B; b += nlb) {
        const Tin* in_row = in + (size_t)b * dim;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < dim; i += MFMA_BLOCK_SIZE) {
            float v = mfma_to_f(in_row[i]);
            smem[i] = v;
            if (resid) resid[(size_t)b * dim + i] = v;
            ss += v * v;
        }
        ss = mfma_block_reduce_sum(ss, red);
        float rstd = rsqrtf(ss / float(dim) + MFMA_RMS_EPS);
        for (int i = threadIdx.x; i < dim; i += MFMA_BLOCK_SIZE)
            mfma_store_as(out[(size_t)b * dim + i], smem[i] * rstd * __bfloat162float(w[i]));
        __syncthreads();
    }
}

// =============================================================================
// QK-norm + RoPE + KV-cache write for this XCD's heads (Q {2x,2x+1}, KV {x}).
// g_q/g_k/g_v are the single shared buffers; this XCD touches only its columns.
// =============================================================================

__device__ void qk_rope_cache_xcd(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_w, const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_table, const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int B, int layer, int num_layers, int position, int max_seq_len, int xcd) {
    int lane  = threadIdx.x % WARP_SIZE;
    int lwarp = mfma_lbid(xcd) * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nlw   = mfma_lnb(xcd) * MFMA_NUM_WARPS;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;
    constexpr int EPR = HEAD_DIM / WARP_SIZE;

    // Q heads: jobs = B * XCD_Q_HEADS, head index = xcd*XCD_Q_HEADS + (job % XCD_Q_HEADS)
    for (int job = lwarp; job < B * XCD_Q_HEADS; job += nlw) {
        int b = job / XCD_Q_HEADS;
        int h = xcd * XCD_Q_HEADS + (job % XCD_Q_HEADS);
        float* q_head = q + (size_t)b * Q_SIZE + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane; i < HEAD_DIM; i += WARP_SIZE) sum_sq += q_head[i] * q_head[i];
        sum_sq = mfma_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(__shfl_sync(WARP_FULL_MASK, sum_sq, 0) / float(HEAD_DIM) + MFMA_RMS_EPS);

        float q_local[EPR];
        #pragma unroll
        for (int i = lane, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            q_local[j] = q_head[i] * scale * __bfloat162float(q_norm_w[i]);
        #pragma unroll
        for (int i = lane, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);
            int pair_idx = i + ((i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2);
            float pair_v = __shfl_sync(WARP_FULL_MASK, q_local[pair_idx / WARP_SIZE], pair_idx % WARP_SIZE);
            q_head[i] = (i < HEAD_DIM / 2) ? (q_local[j] * cos_v - pair_v * sin_v)
                                           : (pair_v * sin_v + q_local[j] * cos_v);
        }
    }

    // K heads (+ V) with cache write: jobs = B * XCD_KV_HEADS, head = xcd (XCD_KV_HEADS==1)
    for (int job = lwarp; job < B * XCD_KV_HEADS; job += nlw) {
        int b = job / XCD_KV_HEADS;
        int h = xcd * XCD_KV_HEADS + (job % XCD_KV_HEADS);
        float* k_head = k + (size_t)b * KV_SIZE + h * HEAD_DIM;
        const float* v_head = v + (size_t)b * KV_SIZE + h * HEAD_DIM;
        size_t kv_bl_base = (((size_t)b * num_layers + layer) * NUM_KV_HEADS + h) * max_seq_len * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + kv_bl_base + (size_t)position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + kv_bl_base + (size_t)position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane; i < HEAD_DIM; i += WARP_SIZE) sum_sq += k_head[i] * k_head[i];
        sum_sq = mfma_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(__shfl_sync(WARP_FULL_MASK, sum_sq, 0) / float(HEAD_DIM) + MFMA_RMS_EPS);

        float k_local[EPR];
        #pragma unroll
        for (int i = lane, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            k_local[j] = k_head[i] * scale * __bfloat162float(k_norm_w[i]);
        #pragma unroll
        for (int i = lane, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(cos_pos[i]);
            float sin_v = __bfloat162float(sin_pos[i]);
            int pair_idx = i + ((i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2);
            float pair_v = __shfl_sync(WARP_FULL_MASK, k_local[pair_idx / WARP_SIZE], pair_idx % WARP_SIZE);
            float k_final = (i < HEAD_DIM / 2) ? (k_local[j] * cos_v - pair_v * sin_v)
                                               : (pair_v * sin_v + k_local[j] * cos_v);
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }
}

// =============================================================================
// Attention for this XCD's Q-heads {2x,2x+1} (KV-head x). One block per job.
// =============================================================================

__device__ void attention_xcd(
    const float* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ attn_out,
    int B, int layer, int num_layers, int cache_len, int max_seq_len, float attn_scale, int xcd) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    __shared__ float s_max_score[MFMA_NUM_WARPS];
    __shared__ float s_sum_exp[MFMA_NUM_WARPS];
    __shared__ float s_out_acc[MFMA_NUM_WARPS][HEAD_DIM];

    int lb = mfma_lbid(xcd), nlb = mfma_lnb(xcd);
    for (int job = lb; job < B * XCD_Q_HEADS; job += nlb) {
        int b  = job / XCD_Q_HEADS;
        int qh = xcd * XCD_Q_HEADS + (job % XCD_Q_HEADS);
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);   // == xcd
        const float* q_head = q + (size_t)b * Q_SIZE + qh * HEAD_DIM;
        __nv_bfloat16* out_head = attn_out + (size_t)b * Q_SIZE + qh * HEAD_DIM;
        size_t kv_base = (((size_t)b * num_layers + layer) * NUM_KV_HEADS + kv_head) * max_seq_len * HEAD_DIM;

        float max_score = -INFINITY, sum_exp = 0.0f;
        float out_acc[HEAD_DIM / WARP_SIZE] = {0.0f};

        for (int pos = warp; pos < cache_len; pos += MFMA_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_base + (size_t)pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_base + (size_t)pos * HEAD_DIM;
            float score = 0.0f;
            for (int d = lane; d < HEAD_DIM; d += WARP_SIZE)
                score += q_head[d] * __bfloat162float(k_pos[d]);
            score = mfma_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(WARP_FULL_MASK, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            float weight = expf(score - max_score);
            sum_exp = sum_exp * exp_diff + weight;
            #pragma unroll
            for (int d = lane, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(v_pos[d]);
        }

        if (lane == 0) { s_max_score[warp] = max_score; s_sum_exp[warp] = sum_exp; }
        #pragma unroll
        for (int d = lane, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) s_out_acc[warp][d] = out_acc[j];
        __syncthreads();

        if (warp == 0) {
            float global_max = -INFINITY;
            for (int w = 0; w < MFMA_NUM_WARPS; w++)
                if (s_max_score[w] > -INFINITY) global_max = fmaxf(global_max, s_max_score[w]);
            float total_sum_exp = 0.0f;
            float final_out[HEAD_DIM / WARP_SIZE] = {0.0f};
            for (int w = 0; w < MFMA_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float sc = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * sc;
                    #pragma unroll
                    for (int d = lane, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                        final_out[j] += s_out_acc[w][d] * sc;
                }
            }
            #pragma unroll
            for (int d = lane, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                out_head[d] = __float2bfloat16(final_out[j] / total_sum_exp);
        }
        __syncthreads();
    }
}

// =============================================================================
// SiLU(gate)*up for this XCD's intermediate slice [x*384,(x+1)*384), all B rows.
// gate/up/out are single shared [Mpad, INT]; this XCD touches only its columns.
// =============================================================================

__device__ void silu_mul_xcd(
    const float* __restrict__ gate, const float* __restrict__ up,
    __nv_bfloat16* __restrict__ out, int B, int Mpad, int xcd) {
    int lt  = mfma_lbid(xcd) * MFMA_BLOCK_SIZE + threadIdx.x;
    int nlt = mfma_lnb(xcd) * MFMA_BLOCK_SIZE;
    int col0 = xcd * XCD_INT;
    // iterate this XCD's B*XCD_INT elements (rows < B only; pad rows stay 0)
    for (int t = lt; t < B * XCD_INT; t += nlt) {
        int b = t / XCD_INT, c = col0 + (t % XCD_INT);
        size_t idx = (size_t)b * INTERMEDIATE_SIZE + c;
        out[idx] = __float2bfloat16(mfma_silu(gate[idx]) * up[idx]);
    }
}

// =============================================================================
// Main XCD-aware batched cooperative kernel
// =============================================================================

__global__ void __launch_bounds__(MFMA_BLOCK_SIZE, 1)
mfma_xcd_decode_kernel(
    const int* __restrict__ input_tokens,          // [B]
    const __nv_bfloat16* __restrict__ embed_weight,
    const MFMALayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden,            // [X, Mpad, HID] bf16 replica
    __nv_bfloat16* __restrict__ g_normalized,      // [X, Mpad, HID] bf16 replica
    float* __restrict__ g_residual,                // [X, Mpad, HID] f32 replica
    float* __restrict__ g_q,                       // [Mpad, Q] f32 (col-sharded)
    float* __restrict__ g_k,                       // [Mpad, KV] f32
    float* __restrict__ g_v,                       // [Mpad, KV] f32
    __nv_bfloat16* __restrict__ g_attn_out,        // [Mpad, Q] bf16 (col-sharded)
    float* __restrict__ g_activations,             // [X, Mpad, HID] f32 replica
    float* __restrict__ g_partial,                 // [X, Mpad, HID] f32 (all-reduce scratch)
    float* __restrict__ g_gate,                    // [Mpad, INT] f32 (col-sharded)
    float* __restrict__ g_up,                      // [Mpad, INT] f32
    __nv_bfloat16* __restrict__ g_mlp,             // [Mpad, INT] bf16 (col-sharded)
    int* __restrict__ bar_arrive,                  // [X]
    int* __restrict__ bar_sense,                   // [X]
    const int* __restrict__ nb_xcd,                // [X]
    int B, int Mpad, int num_layers,
    int position, int cache_len, int max_seq_len, float attn_scale) {
    cg::grid_group grid = cg::this_grid();
    int xcd = blockIdx.x % MQ_NXCD;
    MqXcdBar bar{bar_arrive, bar_sense, nb_xcd};
    bool my_sense = false;

#if MQ_XCD_HIER
    #define MQ_LOCAL_BARRIER() mq_xcd_bar(bar, xcd, my_sense)
#else
    #define MQ_LOCAL_BARRIER() do { (void)bar; (void)my_sense; grid.sync(); } while (0)
#endif

    size_t rep = (size_t)Mpad * HIDDEN_SIZE;

    // Embedding: each XCD fills its OWN hidden replica for all B rows.
    {
        int lt  = mfma_lbid(xcd) * MFMA_BLOCK_SIZE + threadIdx.x;
        int nlt = mfma_lnb(xcd) * MFMA_BLOCK_SIZE;
        __nv_bfloat16* h = hidden + (size_t)xcd * rep;
        for (int t = lt; t < B * HIDDEN_SIZE; t += nlt) {
            int b = t / HIDDEN_SIZE, i = t % HIDDEN_SIZE;
            h[(size_t)b * HIDDEN_SIZE + i] = embed_weight[(size_t)input_tokens[b] * HIDDEN_SIZE + i];
        }
    }
    MQ_LOCAL_BARRIER();

    for (int layer = 0; layer < num_layers; layer++) {
        const MFMALayerWeights& w = layer_weights[layer];
        const __bf16* norm_rep = BF(g_normalized) + (size_t)xcd * rep;

        // 1) input RMSNorm (own replica), save residual
        rmsnorm_xcd(hidden, w.input_layernorm_weight, g_normalized, g_residual, B, Mpad, HIDDEN_SIZE, xcd);
        MQ_LOCAL_BARRIER();

        // 2) QKV expand (output-sharded): this XCD writes its Q/KV column slices
#if MQ_FP8_WEIGHTS
        gemm_xcd_out_fp8<float>(norm_rep, FP8(w.q_proj_weight), w.q_proj_scale, g_q, Mpad, HIDDEN_SIZE, Q_SIZE,
                                xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
        gemm_xcd_out_fp8<float>(norm_rep, FP8(w.k_proj_weight), w.k_proj_scale, g_k, Mpad, HIDDEN_SIZE, KV_SIZE,
                                xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
        gemm_xcd_out_fp8<float>(norm_rep, FP8(w.v_proj_weight), w.v_proj_scale, g_v, Mpad, HIDDEN_SIZE, KV_SIZE,
                                xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
#else
        gemm_xcd_out<float>(norm_rep, BF(w.q_proj_weight), g_q, Mpad, HIDDEN_SIZE, Q_SIZE,
                            xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
        gemm_xcd_out<float>(norm_rep, BF(w.k_proj_weight), g_k, Mpad, HIDDEN_SIZE, KV_SIZE,
                            xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
        gemm_xcd_out<float>(norm_rep, BF(w.v_proj_weight), g_v, Mpad, HIDDEN_SIZE, KV_SIZE,
                            xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
#endif
        MQ_LOCAL_BARRIER();

        // 3) QK-norm + RoPE + KV cache (own heads)
        qk_rope_cache_xcd(g_q, g_k, g_v, w.q_norm_weight, w.k_norm_weight,
                          cos_table, sin_table, k_cache, v_cache,
                          B, layer, num_layers, position, max_seq_len, xcd);
        MQ_LOCAL_BARRIER();

        // 4) attention (own Q-heads, own KV-head cache)
        attention_xcd(g_q, k_cache, v_cache, g_attn_out,
                      B, layer, num_layers, cache_len, max_seq_len, attn_scale, xcd);
        MQ_LOCAL_BARRIER();

        // 5) O-proj CONTRACT (K-sharded) -> per-XCD partial
#if MQ_FP8_WEIGHTS
        gemm_xcd_partial_fp8(BF(g_attn_out), FP8(w.o_proj_weight), w.o_proj_scale, g_partial + (size_t)xcd * rep,
                             Mpad, Q_SIZE, HIDDEN_SIZE, xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
#else
        gemm_xcd_partial(BF(g_attn_out), BF(w.o_proj_weight), g_partial + (size_t)xcd * rep,
                         Mpad, Q_SIZE, HIDDEN_SIZE, xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
#endif
        grid.sync();  // CROSS-XCD: publish all O-proj partials

        // 6) all-reduce partials + pre-attn residual -> g_activations replica
        allreduce_add_resid<float>(g_partial, g_residual, g_activations, Mpad, HIDDEN_SIZE, xcd);
        MQ_LOCAL_BARRIER();

        // 7) post-attn RMSNorm (own replica), save residual
        rmsnorm_xcd(g_activations, w.post_attn_layernorm_weight, g_normalized, g_residual, B, Mpad, HIDDEN_SIZE, xcd);
        MQ_LOCAL_BARRIER();

        // 8) gate + up expand (output-sharded): this XCD's intermediate slice
#if MQ_FP8_WEIGHTS
        gemm_xcd_out_fp8<float>(norm_rep, FP8(w.gate_proj_weight), w.gate_proj_scale, g_gate, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                xcd * XCD_INT, XCD_INT, xcd);
        gemm_xcd_out_fp8<float>(norm_rep, FP8(w.up_proj_weight), w.up_proj_scale, g_up, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                xcd * XCD_INT, XCD_INT, xcd);
#else
        gemm_xcd_out<float>(norm_rep, BF(w.gate_proj_weight), g_gate, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                            xcd * XCD_INT, XCD_INT, xcd);
        gemm_xcd_out<float>(norm_rep, BF(w.up_proj_weight), g_up, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                            xcd * XCD_INT, XCD_INT, xcd);
#endif
        MQ_LOCAL_BARRIER();

        // 9) SiLU(gate)*up (own slice)
        silu_mul_xcd(g_gate, g_up, g_mlp, B, Mpad, xcd);
        MQ_LOCAL_BARRIER();

        // 10) down-proj CONTRACT (K-sharded) -> per-XCD partial
#if MQ_FP8_WEIGHTS
        gemm_xcd_partial_fp8(BF(g_mlp), FP8(w.down_proj_weight), w.down_proj_scale, g_partial + (size_t)xcd * rep,
                             Mpad, INTERMEDIATE_SIZE, HIDDEN_SIZE, xcd * XCD_INT, XCD_INT, xcd);
#else
        gemm_xcd_partial(BF(g_mlp), BF(w.down_proj_weight), g_partial + (size_t)xcd * rep,
                         Mpad, INTERMEDIATE_SIZE, HIDDEN_SIZE, xcd * XCD_INT, XCD_INT, xcd);
#endif
        grid.sync();  // CROSS-XCD: publish all down-proj partials

        // 11) all-reduce partials + post-attn residual -> hidden replica
        allreduce_add_resid<__nv_bfloat16>(g_partial, g_residual, hidden, Mpad, HIDDEN_SIZE, xcd);
        MQ_LOCAL_BARRIER();
    }

    // Final RMSNorm (own replica) -> g_normalized replica (LM head reads replica 0)
    rmsnorm_xcd(hidden, final_norm_weight, g_normalized, nullptr, B, Mpad, HIDDEN_SIZE, xcd);

    #undef MQ_LOCAL_BARRIER
}

// =============================================================================
// LM head (identical to Stage 1): GEMM logits then per-row argmax.
// =============================================================================

__global__ void mfma_xcd_lm_head_gemm(
    const __bf16* __restrict__ A, const __bf16* __restrict__ W,
    float* __restrict__ logits, int Mpad, int in_dim, int out_dim) {
    int warp   = threadIdx.x / WARP_SIZE;
    int gwarp  = blockIdx.x * (MFMA_LM_BLOCK / WARP_SIZE) + warp;
    int nwarp  = gridDim.x * (MFMA_LM_BLOCK / WARP_SIZE);
    int nt     = out_dim / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = gwarp; t < ntiles; t += nwarp) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum(A, W, in_dim, mi * 16, ni * 16, acc);
        mq_mfma_tile_store<float>(logits, out_dim, mi * 16, ni * 16, acc);
    }
}

__global__ void mfma_xcd_lm_head_argmax(
    const float* __restrict__ logits, int* __restrict__ out_tokens, int vocab, int ld) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    int b = blockIdx.x, tid = threadIdx.x;
    const float* row = logits + (size_t)b * ld;
    float mx = -INFINITY; int mi = -1;
    for (int i = tid; i < vocab; i += blockDim.x) { float v = row[i]; if (v > mx) { mx = v; mi = i; } }
    sv[tid] = mx; si[tid] = mi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out_tokens[b] = si[0];
}

// =============================================================================
// In-kernel multi-token decode (per-token overhead lever).
//
// The single-step path launches 3 kernels per token (coop transformer + LM GEMM +
// argmax) and round-trips to the host between tokens (.tolist() D2H sync, next-
// token tensor build, cooperative relaunch of the device-filling grid). At B=1
// that overhead is ~97% of the ~6.9 ms/token wall clock (memory-bound floor is
// ~230 us). mfma_xcd_decode_multi folds the LM head INTO the cooperative kernel
// and loops `num_steps` decode steps on-device: argmax feeds the next embedding
// through a device cur_tokens[B] buffer, position/cache_len advance per step, and
// the whole generation phase is ONE launch with zero host round trips. Greedy
// argmax is deterministic, so the emitted [num_steps,B] tokens are bit-identical
// to the single-step loop (EOS truncation, if any, is a host post-step).
// =============================================================================

// LM head GEMM over the WHOLE cooperative grid (not per-XCD): logits[Mpad,vocab]
// = A[Mpad,in] @ W[vocab,in]^T. Distributes tiles over gridDim.x*MFMA_NUM_WARPS.
__device__ __forceinline__ void lm_head_gemm_grid(
    const __bf16* __restrict__ A, const __bf16* __restrict__ W,
    float* __restrict__ logits, int Mpad, int in_dim, int out_dim) {
    int gwarp = blockIdx.x * MFMA_NUM_WARPS + (threadIdx.x / WARP_SIZE);
    int nwarp = gridDim.x * MFMA_NUM_WARPS;
    int nt    = out_dim / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = gwarp; t < ntiles; t += nwarp) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum(A, W, in_dim, mi * 16, ni * 16, acc);
        mq_mfma_tile_store<float>(logits, out_dim, mi * 16, ni * 16, acc);
    }
}

// Per-row argmax over the grid: block b (b < B) scans the full vocab and writes
// the greedy token to cur_tokens[b] (next-step feedback) and out_step[b] (output).
// Strict '>' + tree reduction that keeps the smaller thread id on ties selects the
// smallest index of the max — same tie-break as mfma_xcd_lm_head_argmax, so the
// 256-thread block here is greedy-identical to the 1024-thread single-step kernel.
__device__ __forceinline__ void lm_argmax_grid(
    const float* __restrict__ logits, int* __restrict__ cur_tokens,
    int* __restrict__ out_step, int B, int vocab) {
    __shared__ float sv[MFMA_BLOCK_SIZE];
    __shared__ int   si[MFMA_BLOCK_SIZE];
    int tid = threadIdx.x;
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        const float* row = logits + (size_t)b * vocab;
        float mx = -INFINITY; int mi = -1;
        for (int i = tid; i < vocab; i += blockDim.x) { float v = row[i]; if (v > mx) { mx = v; mi = i; } }
        sv[tid] = mx; si[tid] = mi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) { cur_tokens[b] = si[0]; out_step[b] = si[0]; }
        __syncthreads();  // reuse sv/si next b iteration
    }
}

__global__ void __launch_bounds__(MFMA_BLOCK_SIZE, 1)
mfma_xcd_decode_multi_kernel(
    const int* __restrict__ input_tokens,          // [B] seed (last prompt token)
    int* __restrict__ cur_tokens,                  // [B] device feedback scratch
    int* __restrict__ output_tokens,               // [num_steps, B] emitted tokens
    const __nv_bfloat16* __restrict__ embed_weight,
    const MFMALayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    __nv_bfloat16* __restrict__ g_attn_out,
    float* __restrict__ g_activations,
    float* __restrict__ g_partial,
    float* __restrict__ g_gate,
    float* __restrict__ g_up,
    __nv_bfloat16* __restrict__ g_mlp,
    float* __restrict__ lm_logits,
    int* __restrict__ bar_arrive,
    int* __restrict__ bar_sense,
    const int* __restrict__ nb_xcd,
    int B, int Mpad, int num_layers,
    int base_position, int base_cache_len, int max_seq_len, float attn_scale,
    int num_steps) {
    cg::grid_group grid = cg::this_grid();
    int xcd = blockIdx.x % MQ_NXCD;
    MqXcdBar bar{bar_arrive, bar_sense, nb_xcd};
    bool my_sense = false;

#if MQ_XCD_HIER
    #define MQ_LOCAL_BARRIER() mq_xcd_bar(bar, xcd, my_sense)
#else
    #define MQ_LOCAL_BARRIER() do { (void)bar; (void)my_sense; grid.sync(); } while (0)
#endif

    size_t rep = (size_t)Mpad * HIDDEN_SIZE;

    // Seed the feedback buffer once from the input tokens.
    {
        int gt = blockIdx.x * blockDim.x + threadIdx.x;
        for (int b = gt; b < B; b += gridDim.x * blockDim.x) cur_tokens[b] = input_tokens[b];
    }
    grid.sync();

    for (int step = 0; step < num_steps; step++) {
        int position  = base_position + step;
        int cache_len = base_cache_len + step;

        // Embedding: each XCD fills its OWN hidden replica for all B rows.
        {
            int lt  = mfma_lbid(xcd) * MFMA_BLOCK_SIZE + threadIdx.x;
            int nlt = mfma_lnb(xcd) * MFMA_BLOCK_SIZE;
            __nv_bfloat16* h = hidden + (size_t)xcd * rep;
            for (int t = lt; t < B * HIDDEN_SIZE; t += nlt) {
                int b = t / HIDDEN_SIZE, i = t % HIDDEN_SIZE;
                h[(size_t)b * HIDDEN_SIZE + i] = embed_weight[(size_t)cur_tokens[b] * HIDDEN_SIZE + i];
            }
        }
        MQ_LOCAL_BARRIER();

        for (int layer = 0; layer < num_layers; layer++) {
            const MFMALayerWeights& w = layer_weights[layer];
            const __bf16* norm_rep = BF(g_normalized) + (size_t)xcd * rep;

            rmsnorm_xcd(hidden, w.input_layernorm_weight, g_normalized, g_residual, B, Mpad, HIDDEN_SIZE, xcd);
            MQ_LOCAL_BARRIER();

#if MQ_FP8_WEIGHTS
            gemm_xcd_out_fp8<float>(norm_rep, FP8(w.q_proj_weight), w.q_proj_scale, g_q, Mpad, HIDDEN_SIZE, Q_SIZE,
                                    xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
            gemm_xcd_out_fp8<float>(norm_rep, FP8(w.k_proj_weight), w.k_proj_scale, g_k, Mpad, HIDDEN_SIZE, KV_SIZE,
                                    xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
            gemm_xcd_out_fp8<float>(norm_rep, FP8(w.v_proj_weight), w.v_proj_scale, g_v, Mpad, HIDDEN_SIZE, KV_SIZE,
                                    xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
#else
            gemm_xcd_out<float>(norm_rep, BF(w.q_proj_weight), g_q, Mpad, HIDDEN_SIZE, Q_SIZE,
                                xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
            gemm_xcd_out<float>(norm_rep, BF(w.k_proj_weight), g_k, Mpad, HIDDEN_SIZE, KV_SIZE,
                                xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
            gemm_xcd_out<float>(norm_rep, BF(w.v_proj_weight), g_v, Mpad, HIDDEN_SIZE, KV_SIZE,
                                xcd * (XCD_KV_HEADS * HEAD_DIM), XCD_KV_HEADS * HEAD_DIM, xcd);
#endif
            MQ_LOCAL_BARRIER();

            qk_rope_cache_xcd(g_q, g_k, g_v, w.q_norm_weight, w.k_norm_weight,
                              cos_table, sin_table, k_cache, v_cache,
                              B, layer, num_layers, position, max_seq_len, xcd);
            MQ_LOCAL_BARRIER();

            attention_xcd(g_q, k_cache, v_cache, g_attn_out,
                          B, layer, num_layers, cache_len, max_seq_len, attn_scale, xcd);
            MQ_LOCAL_BARRIER();

#if MQ_FP8_WEIGHTS
            gemm_xcd_partial_fp8(BF(g_attn_out), FP8(w.o_proj_weight), w.o_proj_scale, g_partial + (size_t)xcd * rep,
                                 Mpad, Q_SIZE, HIDDEN_SIZE, xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
#else
            gemm_xcd_partial(BF(g_attn_out), BF(w.o_proj_weight), g_partial + (size_t)xcd * rep,
                             Mpad, Q_SIZE, HIDDEN_SIZE, xcd * XCD_Q_COLS, XCD_Q_COLS, xcd);
#endif
            grid.sync();  // CROSS-XCD: publish all O-proj partials

            allreduce_add_resid<float>(g_partial, g_residual, g_activations, Mpad, HIDDEN_SIZE, xcd);
            MQ_LOCAL_BARRIER();

            rmsnorm_xcd(g_activations, w.post_attn_layernorm_weight, g_normalized, g_residual, B, Mpad, HIDDEN_SIZE, xcd);
            MQ_LOCAL_BARRIER();

#if MQ_FP8_WEIGHTS
            gemm_xcd_out_fp8<float>(norm_rep, FP8(w.gate_proj_weight), w.gate_proj_scale, g_gate, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                    xcd * XCD_INT, XCD_INT, xcd);
            gemm_xcd_out_fp8<float>(norm_rep, FP8(w.up_proj_weight), w.up_proj_scale, g_up, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                    xcd * XCD_INT, XCD_INT, xcd);
#else
            gemm_xcd_out<float>(norm_rep, BF(w.gate_proj_weight), g_gate, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                xcd * XCD_INT, XCD_INT, xcd);
            gemm_xcd_out<float>(norm_rep, BF(w.up_proj_weight), g_up, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                                xcd * XCD_INT, XCD_INT, xcd);
#endif
            MQ_LOCAL_BARRIER();

            silu_mul_xcd(g_gate, g_up, g_mlp, B, Mpad, xcd);
            MQ_LOCAL_BARRIER();

#if MQ_FP8_WEIGHTS
            gemm_xcd_partial_fp8(BF(g_mlp), FP8(w.down_proj_weight), w.down_proj_scale, g_partial + (size_t)xcd * rep,
                                 Mpad, INTERMEDIATE_SIZE, HIDDEN_SIZE, xcd * XCD_INT, XCD_INT, xcd);
#else
            gemm_xcd_partial(BF(g_mlp), BF(w.down_proj_weight), g_partial + (size_t)xcd * rep,
                             Mpad, INTERMEDIATE_SIZE, HIDDEN_SIZE, xcd * XCD_INT, XCD_INT, xcd);
#endif
            grid.sync();  // CROSS-XCD: publish all down-proj partials

            allreduce_add_resid<__nv_bfloat16>(g_partial, g_residual, hidden, Mpad, HIDDEN_SIZE, xcd);
            MQ_LOCAL_BARRIER();
        }

        // Final RMSNorm (own replica) -> g_normalized; LM head reads replica 0.
        rmsnorm_xcd(hidden, final_norm_weight, g_normalized, nullptr, B, Mpad, HIDDEN_SIZE, xcd);
        grid.sync();  // CROSS-XCD: replica 0 fully written before the LM GEMM reads it

        // LM head (in-kernel): logits then greedy argmax -> next tokens.
        lm_head_gemm_grid(BF(g_normalized), (const __bf16*)lm_head_weight, lm_logits, Mpad, HIDDEN_SIZE, MFMA_VOCAB_SIZE);
        grid.sync();
        lm_argmax_grid(lm_logits, cur_tokens, output_tokens + (size_t)step * B, B, MFMA_VOCAB_SIZE);
        grid.sync();  // cur_tokens visible to every block before the next step's embedding
    }

    #undef MQ_LOCAL_BARRIER
}

// =============================================================================
// Launch
// =============================================================================

// Defined below; the cooperative launch grid MUST equal the block count the
// intra-XCD barrier's nb_xcd[] was built for (see megakernel_batched_xcd.py),
// otherwise the barrier waits on blocks that never launched -> deadlock. Route
// both through this one helper (device-fill for the sharded kernel).
extern "C" int mq_xcd_grid_blocks();
extern "C" int mq_xcd_grid_blocks_for_batch(int B);

extern "C" void launch_mfma_xcd_decode(
    const int* input_tokens, int* output_tokens,
    const void* embed_weight, const MFMALayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden, void* g_normalized, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_activations, void* g_partial,
    void* g_gate, void* g_up, void* g_mlp,
    int* bar_arrive, int* bar_sense, const int* nb_xcd,
    void* lm_logits,
    int B, int Mpad, int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream) {

    void* kernel_args[] = {
        (void*)&input_tokens, (void*)&embed_weight, (void*)&layer_weights,
        (void*)&final_norm_weight, (void*)&cos_table, (void*)&sin_table,
        (void*)&k_cache, (void*)&v_cache, (void*)&hidden, (void*)&g_normalized,
        (void*)&g_residual, (void*)&g_q, (void*)&g_k, (void*)&g_v,
        (void*)&g_attn_out, (void*)&g_activations, (void*)&g_partial,
        (void*)&g_gate, (void*)&g_up, (void*)&g_mlp,
        (void*)&bar_arrive, (void*)&bar_sense, (void*)&nb_xcd,
        (void*)&B, (void*)&Mpad, (void*)&num_layers,
        (void*)&position, (void*)&cache_len, (void*)&max_seq_len, (void*)&attn_scale
    };

    // Same grid the barrier's nb_xcd[] was sized for. Batch-adaptive (fewer blocks at
    // low batch cut barrier cost); the ctor fills nb_xcd for the same B. MQ_GRID_BLOCKS
    // overrides. Using the capped flat-kernel grid here would under-launch every XCD
    // and hang the intra-XCD barrier.
    int grid_blocks = mq_xcd_grid_blocks_for_batch(B);
    cudaLaunchCooperativeKernel((void*)mfma_xcd_decode_kernel, dim3(grid_blocks),
                                dim3(MFMA_BLOCK_SIZE), kernel_args, 0, stream);

    int nt = MFMA_VOCAB_SIZE / 16;
    int ntiles = (Mpad / 16) * nt;
    int lm_blocks = (ntiles + (MFMA_LM_BLOCK / WARP_SIZE) - 1) / (MFMA_LM_BLOCK / WARP_SIZE);
    if (lm_blocks > 8192) lm_blocks = 8192;
    mfma_xcd_lm_head_gemm<<<lm_blocks, MFMA_LM_BLOCK, 0, stream>>>(
        BF(g_normalized), (const __bf16*)lm_head_weight, (float*)lm_logits,
        Mpad, HIDDEN_SIZE, MFMA_VOCAB_SIZE);
    mfma_xcd_lm_head_argmax<<<B, 1024, 0, stream>>>(
        (const float*)lm_logits, output_tokens, MFMA_VOCAB_SIZE, MFMA_VOCAB_SIZE);
}

// Multi-token: ONE cooperative launch runs the whole generation phase on-device.
// No per-token host round trip, no per-token LM-head launches, no cooperative
// relaunch. output_tokens is [num_steps, B]; cur_tokens is [B] device scratch.
extern "C" void launch_mfma_xcd_decode_multi(
    const int* input_tokens, int* cur_tokens, int* output_tokens,
    const void* embed_weight, const MFMALayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden, void* g_normalized, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_activations, void* g_partial,
    void* g_gate, void* g_up, void* g_mlp,
    void* lm_logits,
    int* bar_arrive, int* bar_sense, const int* nb_xcd,
    int B, int Mpad, int num_layers, int base_position, int base_cache_len,
    int max_seq_len, float attn_scale, int num_steps, cudaStream_t stream) {

    void* kernel_args[] = {
        (void*)&input_tokens, (void*)&cur_tokens, (void*)&output_tokens,
        (void*)&embed_weight, (void*)&layer_weights, (void*)&final_norm_weight,
        (void*)&lm_head_weight, (void*)&cos_table, (void*)&sin_table,
        (void*)&k_cache, (void*)&v_cache, (void*)&hidden, (void*)&g_normalized,
        (void*)&g_residual, (void*)&g_q, (void*)&g_k, (void*)&g_v,
        (void*)&g_attn_out, (void*)&g_activations, (void*)&g_partial,
        (void*)&g_gate, (void*)&g_up, (void*)&g_mlp, (void*)&lm_logits,
        (void*)&bar_arrive, (void*)&bar_sense, (void*)&nb_xcd,
        (void*)&B, (void*)&Mpad, (void*)&num_layers,
        (void*)&base_position, (void*)&base_cache_len, (void*)&max_seq_len,
        (void*)&attn_scale, (void*)&num_steps
    };

    int grid_blocks = mq_xcd_grid_blocks_for_batch(B);
    cudaLaunchCooperativeKernel((void*)mfma_xcd_decode_multi_kernel, dim3(grid_blocks),
                                dim3(MFMA_BLOCK_SIZE), kernel_args, 0, stream);
}

// Host helper: fill nb_xcd[x] = #blocks with blockIdx.x % 8 == x for the grid the
// cooperative launch will use (so the intra-XCD barrier knows its arrival target).
extern "C" int mq_xcd_grid_blocks() {
    // Device-fill (one block/CU, rounded to a multiple of 8 XCDs). The sharded
    // kernel scales with grid size where the flat kernel collapses; measured best
    // at device-fill on the batched MFMA path. MQ_GRID_BLOCKS overrides.
    return mq_coop_grid_blocks_fill((void*)mfma_xcd_decode_kernel, MFMA_BLOCK_SIZE, 0, 8);
}

// Batch-adaptive cooperative grid. Profiling (rocprofv3 + grid sweep, MI300X) showed
// this kernel is BARRIER-serialization bound, not HBM/occupancy bound: every extra
// block is another participant in the ~140 grid.sync barriers/token, and barrier cost
// grows with block count. So the throughput-optimal grid shrinks at low batch, where
// there isn't enough MFMA work to amortize the barrier. Measured optima (tok/s peak):
//   B<=4 -> base/4 (72 on 304-CU),  B<=32 -> base/2 (152),  B>=64 -> base (304).
// Encoded as fractions of the device-fill base so it ports to other CU counts. The
// ctor's nb_xcd[] fill and BOTH launch sites must pass the same B, or the intra-XCD
// barrier's arrival target won't match the launched grid and it will deadlock.
extern "C" int mq_xcd_grid_blocks_for_batch(int B) {
    if (const char* env = getenv("MQ_GRID_BLOCKS")) { int v = atoi(env); if (v > 0) return v; }
    static int base = mq_xcd_grid_blocks();     // device-fill, already a multiple of 8
    int g = base;
    if (B <= 4)       g = base / 4;
    else if (B <= 32) g = base / 2;
    g = (g / MQ_NXCD) * MQ_NXCD;                 // keep XCD-balanced (blockIdx.x % 8)
    if (g < MQ_NXCD) g = MQ_NXCD;
    return g;
}
