/**
 * Batched fused decode with MFMA GEMM projections (MI300X / gfx942).
 *
 * Stage 1 of the MI300X optimization roadmap. Where fused_decode_ldg.cu is a
 * batch-1, warp-per-row GEMV megakernel (HBM-bandwidth-bound: re-streams every
 * weight per token, matrix cores idle), this variant processes B sequences at
 * once and turns each projection into an MFMA bf16 GEMM tile:
 *
 *     C[B, out] = A[B, in] @ W[out, in]^T
 *
 * A weight tile is loaded once and applied to all B rows, so per-token weight
 * HBM traffic drops ~B× and the 16x16x16 matrix cores (mfma.cuh) do the work.
 * Keep fused_decode_ldg.cu as the A/B baseline.
 *
 * Assumption (Stage 1): LOCKSTEP batched decode — all B sequences share the
 * same `position` / `cache_len` (the standard fixed-length throughput
 * benchmark). Per-sequence variable length (continuous batching) is a later
 * refinement; the KV cache is already laid out per-sequence to allow it.
 *
 * Buffer dtype split:
 *   - MFMA A-inputs (g_normalized, g_attn_out, g_mlp_intermediate, hidden) are
 *     bf16 [Mpad, dim]; produced by the preceding norm/elementwise phase.
 *   - RoPE-precision / accumulation buffers (g_q/g_k/g_v, g_residual,
 *     g_activations, g_gate, g_up) are f32.
 *   - M (batch) is padded to Mpad = round_up(B, 16). Pad rows stay zero (A
 *     buffers allocated zeroed, per-row phases only touch rows < B), so padded
 *     tiles contribute clean zeros and are never read downstream.
 */

#include "config.cuh"
#include "mfma.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int MFMA_BLOCK_SIZE = 256;
constexpr int MFMA_NUM_WARPS  = MFMA_BLOCK_SIZE / WARP_SIZE;   // 4 on gfx942
constexpr float MFMA_RMS_EPS  = 1e-6f;

constexpr int MFMA_VOCAB_SIZE = 151936;
constexpr int MFMA_LM_BLOCK   = 256;

struct MFMALayerWeights {
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
};

// Reinterpret a Qwen bf16 weight/activation pointer (__nv_bfloat16 == the HIP
// bf16 struct) as the clang builtin __bf16 the MFMA tile helpers consume. Same
// 2-byte layout — a pointer reinterpret is well-defined.
#define BF(p) reinterpret_cast<const __bf16*>(p)

// =============================================================================
// Small helpers
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

// Block-wide sum reduction; `red` is __shared__ float[MFMA_NUM_WARPS]. Result
// broadcast to all threads. Caller must __syncthreads() before reusing `red`.
__device__ __forceinline__ float mfma_block_reduce_sum(float v, float* red) {
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    v = mfma_warp_reduce_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = (threadIdx.x < MFMA_NUM_WARPS) ? red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        s = mfma_warp_reduce_sum(s);
        if (lane == 0) red[0] = s;
    }
    __syncthreads();
    return red[0];
}

// =============================================================================
// MFMA GEMM projection: C[Mpad,out] = A[Mpad,in] @ W[out,in]^T.
// Tiles distributed over the whole cooperative grid's wavefronts. No grid.sync
// inside — the caller places the barrier. If ADD_RESID, fuse C += resid[Mpad,out].
// =============================================================================

template <typename Tout, bool ADD_RESID>
__device__ __forceinline__ void mfma_gemm(
    const __bf16* __restrict__ A, const __bf16* __restrict__ W,
    Tout* __restrict__ C, const float* __restrict__ resid,
    int Mpad, int in_dim, int out_dim) {
    int warp   = threadIdx.x / WARP_SIZE;
    int gwarp  = blockIdx.x * MFMA_NUM_WARPS + warp;
    int nwarp  = gridDim.x * MFMA_NUM_WARPS;
    int nt     = out_dim / 16;
    int ntiles = (Mpad / 16) * nt;
    for (int t = gwarp; t < ntiles; t += nwarp) {
        int mi = t / nt, ni = t % nt;
        mq_f32x4 acc = {0.f, 0.f, 0.f, 0.f};
        mq_mfma_tile_accum(A, W, in_dim, mi * 16, ni * 16, acc);
        if (ADD_RESID) mq_mfma_tile_store_add<Tout>(C, resid, out_dim, mi * 16, ni * 16, acc);
        else           mq_mfma_tile_store<Tout>(C, out_dim, mi * 16, ni * 16, acc);
    }
}

// =============================================================================
// Per-row RMSNorm over B rows (one block per row, striding). Reads in[b,dim],
// writes normalized out[b,dim]; optionally saves the pre-norm value to resid.
// =============================================================================

template <typename Tin, typename Tout>
__device__ void mfma_rmsnorm(
    const Tin* __restrict__ in, const __nv_bfloat16* __restrict__ w,
    Tout* __restrict__ out, float* __restrict__ resid, int B, int dim) {
    __shared__ float smem[HIDDEN_SIZE];
    __shared__ float red[MFMA_NUM_WARPS];
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
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
        for (int i = threadIdx.x; i < dim; i += MFMA_BLOCK_SIZE) {
            mfma_store_as(out[(size_t)b * dim + i], smem[i] * rstd * __bfloat162float(w[i]));
        }
        __syncthreads();  // protect smem/red before next row
    }
}

// =============================================================================
// Per-(b, head) QK-norm + RoPE + KV-cache write. Warp-per-head, flattened jobs.
// =============================================================================

__device__ void mfma_qk_norm_rope_cache(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_w, const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_table, const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int B, int layer, int num_layers, int position, int max_seq_len) {
    int warp  = threadIdx.x / WARP_SIZE;
    int lane  = threadIdx.x % WARP_SIZE;
    int gwarp = blockIdx.x * MFMA_NUM_WARPS + warp;
    int nwarp = gridDim.x * MFMA_NUM_WARPS;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;
    constexpr int EPR = HEAD_DIM / WARP_SIZE;  // elems/lane (2 on gfx942)

    // --- Q heads ---
    for (int job = gwarp; job < B * NUM_Q_HEADS; job += nwarp) {
        int b = job / NUM_Q_HEADS, h = job % NUM_Q_HEADS;
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

    // --- K heads (+ V) with cache write ---
    size_t kv_bl_base = ((size_t)0);  // computed per b below
    for (int job = gwarp; job < B * NUM_KV_HEADS; job += nwarp) {
        int b = job / NUM_KV_HEADS, h = job % NUM_KV_HEADS;
        float* k_head = k + (size_t)b * KV_SIZE + h * HEAD_DIM;
        const float* v_head = v + (size_t)b * KV_SIZE + h * HEAD_DIM;
        // cache index (b, layer, h, position, :)
        kv_bl_base = (((size_t)b * num_layers + layer) * NUM_KV_HEADS + h) * max_seq_len * HEAD_DIM;
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
// Per-(b, q-head) attention, online softmax. One block per job (warps split
// cache positions), striding over B*NUM_Q_HEADS jobs. Writes bf16 attn_out.
// =============================================================================

__device__ void mfma_attention(
    const float* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, __nv_bfloat16* __restrict__ attn_out,
    int B, int layer, int num_layers, int cache_len, int max_seq_len, float attn_scale) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    __shared__ float s_max_score[MFMA_NUM_WARPS];
    __shared__ float s_sum_exp[MFMA_NUM_WARPS];
    __shared__ float s_out_acc[MFMA_NUM_WARPS][HEAD_DIM];

    for (int job = blockIdx.x; job < B * NUM_Q_HEADS; job += gridDim.x) {
        int b = job / NUM_Q_HEADS, qh = job % NUM_Q_HEADS;
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float* q_head = q + (size_t)b * Q_SIZE + qh * HEAD_DIM;
        __nv_bfloat16* out_head = attn_out + (size_t)b * Q_SIZE + qh * HEAD_DIM;
        // KV cache base for (b, layer, kv_head)
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
        __syncthreads();  // protect shared before next job
    }
}

// =============================================================================
// Elementwise SiLU(gate) * up over B*INTERMEDIATE_SIZE, writing bf16 g_mlp.
// =============================================================================

__device__ void mfma_silu_mul(
    const float* __restrict__ gate, const float* __restrict__ up,
    __nv_bfloat16* __restrict__ out, int B) {
    size_t total = (size_t)B * INTERMEDIATE_SIZE;
    size_t gtid = (size_t)blockIdx.x * MFMA_BLOCK_SIZE + threadIdx.x;
    size_t stride = (size_t)gridDim.x * MFMA_BLOCK_SIZE;
    for (size_t t = gtid; t < total; t += stride)
        out[t] = __float2bfloat16(mfma_silu(gate[t]) * up[t]);
}

// =============================================================================
// Main batched cooperative kernel
// =============================================================================

__global__ void __launch_bounds__(MFMA_BLOCK_SIZE, 1)
mfma_decode_kernel(
    const int* __restrict__ input_tokens,          // [B]
    const __nv_bfloat16* __restrict__ embed_weight,
    const MFMALayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,           // [B, layers, kv_heads, max_seq, head_dim]
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden,            // [Mpad, HID] bf16 (A for QKV / layer state)
    __nv_bfloat16* __restrict__ g_normalized,      // [Mpad, HID] bf16 (norm out / postnorm out / final)
    float* __restrict__ g_residual,                // [Mpad, HID] f32
    float* __restrict__ g_q,                       // [Mpad, Q] f32
    float* __restrict__ g_k,                       // [Mpad, KV] f32
    float* __restrict__ g_v,                       // [Mpad, KV] f32
    __nv_bfloat16* __restrict__ g_attn_out,        // [Mpad, Q] bf16 (A for O-proj)
    float* __restrict__ g_activations,             // [Mpad, HID] f32 (O-proj + resid)
    float* __restrict__ g_gate,                    // [Mpad, INT] f32
    float* __restrict__ g_up,                      // [Mpad, INT] f32
    __nv_bfloat16* __restrict__ g_mlp,             // [Mpad, INT] bf16 (A for down-proj)
    int B, int Mpad, int num_layers,
    int position, int cache_len, int max_seq_len, float attn_scale) {
    cg::grid_group grid = cg::this_grid();

    // Embedding lookup: hidden[b, :] = embed[token[b], :]
    for (size_t t = (size_t)blockIdx.x * MFMA_BLOCK_SIZE + threadIdx.x;
         t < (size_t)B * HIDDEN_SIZE; t += (size_t)gridDim.x * MFMA_BLOCK_SIZE) {
        int b = t / HIDDEN_SIZE, i = t % HIDDEN_SIZE;
        hidden[(size_t)b * HIDDEN_SIZE + i] = embed_weight[(size_t)input_tokens[b] * HIDDEN_SIZE + i];
    }
    grid.sync();

    for (int layer = 0; layer < num_layers; layer++) {
        const MFMALayerWeights& w = layer_weights[layer];

        // 1) input RMSNorm  (hidden -> g_normalized bf16, save residual)
        mfma_rmsnorm(hidden, w.input_layernorm_weight, g_normalized, g_residual, B, HIDDEN_SIZE);
        grid.sync();

        // 2) QKV projections (GEMM)
        mfma_gemm<float, false>(BF(g_normalized), BF(w.q_proj_weight), g_q, nullptr, Mpad, HIDDEN_SIZE, Q_SIZE);
        mfma_gemm<float, false>(BF(g_normalized), BF(w.k_proj_weight), g_k, nullptr, Mpad, HIDDEN_SIZE, KV_SIZE);
        mfma_gemm<float, false>(BF(g_normalized), BF(w.v_proj_weight), g_v, nullptr, Mpad, HIDDEN_SIZE, KV_SIZE);
        grid.sync();

        // 3) QK-norm + RoPE + KV cache
        mfma_qk_norm_rope_cache(g_q, g_k, g_v, w.q_norm_weight, w.k_norm_weight,
                                cos_table, sin_table, k_cache, v_cache,
                                B, layer, num_layers, position, max_seq_len);
        grid.sync();

        // 4) attention (-> g_attn_out bf16)
        mfma_attention(g_q, k_cache, v_cache, g_attn_out,
                       B, layer, num_layers, cache_len, max_seq_len, attn_scale);
        grid.sync();

        // 5) O-proj (GEMM) + pre-attn residual -> g_activations f32
        mfma_gemm<float, true>(BF(g_attn_out), BF(w.o_proj_weight), g_activations, g_residual,
                               Mpad, Q_SIZE, HIDDEN_SIZE);
        grid.sync();

        // 6) post-attn RMSNorm (g_activations -> g_normalized bf16, save residual)
        mfma_rmsnorm(g_activations, w.post_attn_layernorm_weight, g_normalized, g_residual, B, HIDDEN_SIZE);
        grid.sync();

        // 7) gate + up projections (GEMM)
        mfma_gemm<float, false>(BF(g_normalized), BF(w.gate_proj_weight), g_gate, nullptr, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        mfma_gemm<float, false>(BF(g_normalized), BF(w.up_proj_weight),   g_up,   nullptr, Mpad, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        grid.sync();

        // 8) SiLU(gate) * up -> g_mlp bf16
        mfma_silu_mul(g_gate, g_up, g_mlp, B);
        grid.sync();

        // 9) down-proj (GEMM) + post-attn residual -> hidden bf16
        mfma_gemm<__nv_bfloat16, true>(BF(g_mlp), BF(w.down_proj_weight), hidden, g_residual,
                                       Mpad, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        grid.sync();
    }

    // Final RMSNorm (hidden -> g_normalized bf16, ready as MFMA A for LM head)
    mfma_rmsnorm(hidden, final_norm_weight, g_normalized, nullptr, B, HIDDEN_SIZE);
}

// =============================================================================
// LM head: batched GEMM logits[Mpad,VOCAB] = g_normalized @ lm_weight^T, then
// per-row argmax.
// =============================================================================

__global__ void mfma_lm_head_gemm(
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

__global__ void mfma_lm_head_argmax(
    const float* __restrict__ logits, int* __restrict__ out_tokens, int vocab, int ld) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    int b = blockIdx.x, tid = threadIdx.x;
    const float* row = logits + (size_t)b * ld;

    float mx = -INFINITY; int mi = -1;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = row[i];
        if (v > mx) { mx = v; mi = i; }
    }
    sv[tid] = mx; si[tid] = mi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out_tokens[b] = si[0];
}

// =============================================================================
// Launch function
// =============================================================================

extern "C" void launch_mfma_decode(
    const int* input_tokens,          // device [B]
    int* output_tokens,               // device [B]
    const void* embed_weight,
    const MFMALayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden,
    void* g_normalized,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_activations,
    void* g_gate,
    void* g_up,
    void* g_mlp,
    void* lm_logits,
    int B,
    int Mpad,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    void* kernel_args[] = {
        (void*)&input_tokens, (void*)&embed_weight, (void*)&layer_weights,
        (void*)&final_norm_weight, (void*)&cos_table, (void*)&sin_table,
        (void*)&k_cache, (void*)&v_cache, (void*)&hidden, (void*)&g_normalized,
        (void*)&g_residual, (void*)&g_q, (void*)&g_k, (void*)&g_v,
        (void*)&g_attn_out, (void*)&g_activations, (void*)&g_gate, (void*)&g_up,
        (void*)&g_mlp, (void*)&B, (void*)&Mpad, (void*)&num_layers,
        (void*)&position, (void*)&cache_len, (void*)&max_seq_len, (void*)&attn_scale
    };

    static int grid_blocks = mq_coop_grid_blocks((void*)mfma_decode_kernel, MFMA_BLOCK_SIZE, 0);
    cudaLaunchCooperativeKernel((void*)mfma_decode_kernel, dim3(grid_blocks),
                                dim3(MFMA_BLOCK_SIZE), kernel_args, 0, stream);

    // LM head GEMM -> per-row argmax
    int nt = MFMA_VOCAB_SIZE / 16;
    int ntiles = (Mpad / 16) * nt;
    int lm_blocks = (ntiles + (MFMA_LM_BLOCK / WARP_SIZE) - 1) / (MFMA_LM_BLOCK / WARP_SIZE);
    if (lm_blocks > 8192) lm_blocks = 8192;
    mfma_lm_head_gemm<<<lm_blocks, MFMA_LM_BLOCK, 0, stream>>>(
        BF(g_normalized), (const __bf16*)lm_head_weight, (float*)lm_logits,
        Mpad, HIDDEN_SIZE, MFMA_VOCAB_SIZE);
    mfma_lm_head_argmax<<<B, 1024, 0, stream>>>(
        (const float*)lm_logits, output_tokens, MFMA_VOCAB_SIZE, MFMA_VOCAB_SIZE);
}
