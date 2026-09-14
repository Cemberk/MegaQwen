"""
Stage 2 — software XCD-aware batched MFMA megakernel decode (MI300X / gfx942).

Wraps csrc/megakernel/fused_decode_mfma_xcd.cu. Same math as the Stage-1 batched
MFMA kernel (megakernel_batched.py), but the transformer is mapped onto the 8 XCD
chiplets like an in-kernel tensor-parallel layout so that almost every phase
boundary becomes an INTRA-XCD barrier; a full cross-XCD grid.sync() survives only
at the two genuine all-reduces (O-proj + down-proj partial sums).

Extra buffers vs Stage 1:
  - Per-XCD REPLICAS [X, Mpad, dim] of the residual-stream buffers (hidden,
    g_normalized, g_residual, g_activations) — each XCD reads/writes only its own
    replica so norm/elementwise stay intra-XCD.
  - g_partial [X, Mpad, HID] — per-XCD contract-projection partials, all-reduced.
  - bar_arrive[X], bar_sense[X], nb_xcd[X] — sense-reversing intra-XCD barrier
    state; nb_xcd[x] = #blocks with blockIdx.x % 8 == x for the cooperative grid.

Compile switch MQ_XCD_HIER: 0 -> every barrier is grid.sync() (validate the
sharding + all-reduce independent of the hierarchical barrier); 1 -> the 9 local
phase boundaries use the intra-XCD barrier (the actual Stage-2 win). Two variants
can coexist (distinct extension names) so a single process can A/B them.
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from build_flags import cuda_cflags  # noqa: E402

from megakernel_decode import (  # noqa: E402
    load_qwen3_weights,
    HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
    HEAD_DIM, Q_SIZE, KV_SIZE, NUM_LAYERS, VOCAB_SIZE,
)

MQ_NXCD = 8

_xcd_kernels = {}  # hier(0/1) -> compiled module


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_xcd_kernel(hier: int = 1):
    if hier in _xcd_kernels:
        return _xcd_kernels[hier]

    # MFMA builtin is CDNA-only; pin gfx942 or load_inline builds every ROCm arch.
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

    cuda_src = _get_cuda_source("fused_decode_mfma_xcd.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define MQ_NXCD 8

struct MFMALayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

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
    float attn_scale, cudaStream_t stream
);

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
    int max_seq_len, float attn_scale, int num_steps, cudaStream_t stream
);

extern "C" int mq_xcd_grid_blocks();
extern "C" int mq_xcd_grid_blocks_for_batch(int B);

static inline int round_up16(int x) { return ((x + 15) / 16) * 16; }

class MegakernelXcdDecoder {
public:
    MegakernelXcdDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len,
        int batch
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), batch_(batch) {

        Mpad_ = round_up16(batch_);

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        layer_weights_tensors_ = layer_weights_flat;

        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight      = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight               = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight               = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight               = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight               = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight               = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight               = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight  = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight            = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight              = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight            = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(MFMALayerWeights)},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(MFMALayerWeights), cudaMemcpyHostToDevice);

        auto bf16 = torch::dtype(torch::kBFloat16).device(torch::kCUDA);
        auto f32  = torch::dtype(torch::kFloat32).device(torch::kCUDA);
        auto i32  = torch::dtype(torch::kInt32).device(torch::kCUDA);

        k_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);
        v_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);

        // Per-XCD REPLICAS of the residual-stream buffers: [X, Mpad, HID].
        hidden_       = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, bf16);
        g_normalized_ = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, bf16);
        g_residual_   = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);
        g_activations_= torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);
        g_partial_    = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);

        // Single shared column-sharded buffers (disjoint per-XCD column slices).
        g_q_          = torch::zeros({Mpad_, Q_SIZE}, f32);
        g_k_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_v_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_attn_out_   = torch::zeros({Mpad_, Q_SIZE}, bf16);
        g_gate_       = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_up_         = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_mlp_        = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, bf16);
        lm_logits_    = torch::zeros({Mpad_, VOCAB_SIZE}, f32);
        output_tokens_= torch::zeros({batch_}, i32);
        cur_tokens_   = torch::zeros({batch_}, i32);  // in-kernel multi-step feedback

        // Intra-XCD barrier state. nb_xcd[x] = #blocks with blockIdx.x % 8 == x
        // for the cooperative grid this kernel launches on.
        bar_arrive_ = torch::zeros({MQ_NXCD}, i32);
        bar_sense_  = torch::zeros({MQ_NXCD}, i32);
        int G = mq_xcd_grid_blocks_for_batch(batch_);
        int nb_host[MQ_NXCD];
        for (int x = 0; x < MQ_NXCD; x++) nb_host[x] = (G - x + MQ_NXCD - 1) / MQ_NXCD;
        nb_xcd_ = torch::empty({MQ_NXCD}, i32);
        cudaMemcpy(nb_xcd_.data_ptr(), nb_host, MQ_NXCD * sizeof(int), cudaMemcpyHostToDevice);

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf((float)HEAD_DIM);
    }

    static const int HIDDEN_SIZE = 1024;
    static const int INTERMEDIATE_SIZE = 3072;
    static const int NUM_KV_HEADS = 8;
    static const int HEAD_DIM = 128;
    static const int Q_SIZE = 16 * 128;
    static const int KV_SIZE = 8 * 128;
    static const int VOCAB_SIZE = 151936;

    torch::Tensor decode_step(torch::Tensor tokens) {
        TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
        TORCH_CHECK(tokens.numel() == batch_, "tokens must have batch elements");
        auto tok = tokens.to(torch::kCUDA).contiguous();

        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        // Clean barrier state each launch (async on the stream, no host sync):
        // an odd number of barrier rounds would otherwise leave sense flipped.
        bar_arrive_.zero_();
        bar_sense_.zero_();

        launch_mfma_xcd_decode(
            (const int*)tok.data_ptr(),
            (int*)output_tokens_.data_ptr(),
            embed_weight_.data_ptr(),
            (const MFMALayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_.data_ptr(),
            g_normalized_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_activations_.data_ptr(),
            g_partial_.data_ptr(),
            g_gate_.data_ptr(),
            g_up_.data_ptr(),
            g_mlp_.data_ptr(),
            (int*)bar_arrive_.data_ptr(),
            (int*)bar_sense_.data_ptr(),
            (const int*)nb_xcd_.data_ptr(),
            lm_logits_.data_ptr(),
            batch_, Mpad_, num_layers_, position_, cache_len, max_seq_len_,
            attn_scale_, stream
        );

        position_++;
        return output_tokens_.clone();
    }

    // In-kernel multi-token decode: ONE cooperative launch emits num_steps tokens
    // (greedy) with zero host round trips. Returns [num_steps, batch] int32 on CUDA.
    torch::Tensor decode_multi(torch::Tensor tokens, int num_steps) {
        TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
        TORCH_CHECK(tokens.numel() == batch_, "tokens must have batch elements");
        TORCH_CHECK(num_steps >= 1, "num_steps must be >= 1");
        auto tok = tokens.to(torch::kCUDA).contiguous();

        auto i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
        auto out = torch::empty({num_steps, batch_}, i32);

        int base_cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        // One reset for the whole run; the sense-reversing barrier self-alternates
        // across every step's identical barrier rounds (no per-step reset needed).
        bar_arrive_.zero_();
        bar_sense_.zero_();

        launch_mfma_xcd_decode_multi(
            (const int*)tok.data_ptr(),
            (int*)cur_tokens_.data_ptr(),
            (int*)out.data_ptr(),
            embed_weight_.data_ptr(),
            (const MFMALayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_.data_ptr(),
            g_normalized_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_activations_.data_ptr(),
            g_partial_.data_ptr(),
            g_gate_.data_ptr(),
            g_up_.data_ptr(),
            g_mlp_.data_ptr(),
            lm_logits_.data_ptr(),
            (int*)bar_arrive_.data_ptr(),
            (int*)bar_sense_.data_ptr(),
            (const int*)nb_xcd_.data_ptr(),
            batch_, Mpad_, num_layers_, position_, base_cache_len, max_seq_len_,
            attn_scale_, num_steps, stream
        );

        position_ += num_steps;
        return out;
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }
    int batch() const { return batch_; }

private:
    int num_layers_, max_seq_len_, batch_, Mpad_, position_;
    float attn_scale_;

    torch::Tensor embed_weight_, final_norm_weight_, lm_head_weight_;
    torch::Tensor cos_table_, sin_table_, d_layer_weights_;
    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<MFMALayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_, g_normalized_, g_residual_, g_activations_, g_partial_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_gate_, g_up_, g_mlp_, lm_logits_, output_tokens_, cur_tokens_;
    torch::Tensor bar_arrive_, bar_sense_, nb_xcd_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelXcdDecoder>(m, "MegakernelXcdDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>())
        .def("decode_step", &MegakernelXcdDecoder::decode_step)
        .def("decode_multi", &MegakernelXcdDecoder::decode_multi)
        .def("reset", &MegakernelXcdDecoder::reset)
        .def("position", &MegakernelXcdDecoder::position)
        .def("batch", &MegakernelXcdDecoder::batch);
}
"""

    # pybind11's type registry is keyed on the C++ type and is process-global, so
    # two variants that share the class name collide ("already registered") once
    # both are imported into one process. Give each hier its own class name.
    cls_name = f"MegakernelXcdDecoder_h{hier}"
    cpp_src = cpp_src.replace("MegakernelXcdDecoder", cls_name)

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    mod = load_inline(
        name=f"megakernel_batched_xcd_h{hier}",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=cuda_cflags(
            include_dirs=[kernel_dir],
            nvcc_extra=["-arch=sm_86", f"-DMQ_XCD_HIER={hier}"],
        ),
        verbose=False,
    )
    _xcd_kernels[hier] = mod
    return mod


# =============================================================================
# Weight-only fp8 (W8A16) variant: same XCD kernel compiled with
# -DMQ_FP8_WEIGHTS=1. The 7 projection weights are stored e4m3fnuz + a
# per-output-channel f32 scale; activations + MFMA compute stay bf16. Decode is
# HBM-bound, so halving projection weight bytes is the throughput lever. Accuracy
# validated by probe_fp8_accuracy.py (40/40 identical greedy vs bf16).
# =============================================================================

_xcd_fp8_kernels = {}  # hier(0/1) -> compiled fp8 module

FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX = 240.0
# Slots within each 11-tensor layer block that are 2-D projections (quant targets),
# in the fixed order q,k,v,o,gate,up,down -> scale slots 0..6.
_PROJ_SLOTS = (1, 2, 3, 6, 8, 9, 10)


def _compile_xcd_fp8_kernel(hier: int = 1):
    if hier in _xcd_fp8_kernels:
        return _xcd_fp8_kernels[hier]

    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
    cuda_src = _get_cuda_source("fused_decode_mfma_xcd.cu")

    # Host struct MUST match the device struct compiled with MQ_FP8_WEIGHTS=1:
    # 11 weight pointers followed by 7 per-output-channel scale pointers.
    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define MQ_NXCD 8

struct MFMALayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
    const float* q_proj_scale;
    const float* k_proj_scale;
    const float* v_proj_scale;
    const float* o_proj_scale;
    const float* gate_proj_scale;
    const float* up_proj_scale;
    const float* down_proj_scale;
};

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
    float attn_scale, cudaStream_t stream
);

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
    int max_seq_len, float attn_scale, int num_steps, cudaStream_t stream
);

extern "C" int mq_xcd_grid_blocks();
extern "C" int mq_xcd_grid_blocks_for_batch(int B);

static inline int round_up16(int x) { return ((x + 15) / 16) * 16; }

class MegakernelXcdFp8Decoder {
public:
    MegakernelXcdFp8Decoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        std::vector<torch::Tensor> scale_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor lm_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len,
        int batch
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len), batch_(batch) {

        Mpad_ = round_up16(batch_);

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        layer_weights_tensors_ = layer_weights_flat;
        scale_tensors_ = scale_flat;

        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight      = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight               = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight               = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight               = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight               = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight               = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight               = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight  = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight            = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight              = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight            = layer_weights_flat[i * 11 + 10].data_ptr();
            // 7 per-output-channel dequant scales, order q,k,v,o,gate,up,down.
            layer_weights_[i].q_proj_scale    = (const float*)scale_flat[i * 7 + 0].data_ptr();
            layer_weights_[i].k_proj_scale    = (const float*)scale_flat[i * 7 + 1].data_ptr();
            layer_weights_[i].v_proj_scale    = (const float*)scale_flat[i * 7 + 2].data_ptr();
            layer_weights_[i].o_proj_scale    = (const float*)scale_flat[i * 7 + 3].data_ptr();
            layer_weights_[i].gate_proj_scale = (const float*)scale_flat[i * 7 + 4].data_ptr();
            layer_weights_[i].up_proj_scale   = (const float*)scale_flat[i * 7 + 5].data_ptr();
            layer_weights_[i].down_proj_scale = (const float*)scale_flat[i * 7 + 6].data_ptr();
        }

        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(MFMALayerWeights)},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(MFMALayerWeights), cudaMemcpyHostToDevice);

        auto bf16 = torch::dtype(torch::kBFloat16).device(torch::kCUDA);
        auto f32  = torch::dtype(torch::kFloat32).device(torch::kCUDA);
        auto i32  = torch::dtype(torch::kInt32).device(torch::kCUDA);

        k_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);
        v_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);

        hidden_       = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, bf16);
        g_normalized_ = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, bf16);
        g_residual_   = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);
        g_activations_= torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);
        g_partial_    = torch::zeros({MQ_NXCD, Mpad_, HIDDEN_SIZE}, f32);

        g_q_          = torch::zeros({Mpad_, Q_SIZE}, f32);
        g_k_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_v_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_attn_out_   = torch::zeros({Mpad_, Q_SIZE}, bf16);
        g_gate_       = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_up_         = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_mlp_        = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, bf16);
        lm_logits_    = torch::zeros({Mpad_, VOCAB_SIZE}, f32);
        output_tokens_= torch::zeros({batch_}, i32);
        cur_tokens_   = torch::zeros({batch_}, i32);  // in-kernel multi-step feedback

        bar_arrive_ = torch::zeros({MQ_NXCD}, i32);
        bar_sense_  = torch::zeros({MQ_NXCD}, i32);
        int G = mq_xcd_grid_blocks();
        int nb_host[MQ_NXCD];
        for (int x = 0; x < MQ_NXCD; x++) nb_host[x] = (G - x + MQ_NXCD - 1) / MQ_NXCD;
        nb_xcd_ = torch::empty({MQ_NXCD}, i32);
        cudaMemcpy(nb_xcd_.data_ptr(), nb_host, MQ_NXCD * sizeof(int), cudaMemcpyHostToDevice);

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf((float)HEAD_DIM);
    }

    static const int HIDDEN_SIZE = 1024;
    static const int INTERMEDIATE_SIZE = 3072;
    static const int NUM_KV_HEADS = 8;
    static const int HEAD_DIM = 128;
    static const int Q_SIZE = 16 * 128;
    static const int KV_SIZE = 8 * 128;
    static const int VOCAB_SIZE = 151936;

    torch::Tensor decode_step(torch::Tensor tokens) {
        TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
        TORCH_CHECK(tokens.numel() == batch_, "tokens must have batch elements");
        auto tok = tokens.to(torch::kCUDA).contiguous();

        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        bar_arrive_.zero_();
        bar_sense_.zero_();

        launch_mfma_xcd_decode(
            (const int*)tok.data_ptr(),
            (int*)output_tokens_.data_ptr(),
            embed_weight_.data_ptr(),
            (const MFMALayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_.data_ptr(),
            g_normalized_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_activations_.data_ptr(),
            g_partial_.data_ptr(),
            g_gate_.data_ptr(),
            g_up_.data_ptr(),
            g_mlp_.data_ptr(),
            (int*)bar_arrive_.data_ptr(),
            (int*)bar_sense_.data_ptr(),
            (const int*)nb_xcd_.data_ptr(),
            lm_logits_.data_ptr(),
            batch_, Mpad_, num_layers_, position_, cache_len, max_seq_len_,
            attn_scale_, stream
        );

        position_++;
        return output_tokens_.clone();
    }

    // In-kernel multi-token decode: ONE cooperative launch emits num_steps tokens
    // (greedy) with zero host round trips. Returns [num_steps, batch] int32 on CUDA.
    torch::Tensor decode_multi(torch::Tensor tokens, int num_steps) {
        TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
        TORCH_CHECK(tokens.numel() == batch_, "tokens must have batch elements");
        TORCH_CHECK(num_steps >= 1, "num_steps must be >= 1");
        auto tok = tokens.to(torch::kCUDA).contiguous();

        auto i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
        auto out = torch::empty({num_steps, batch_}, i32);

        int base_cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        bar_arrive_.zero_();
        bar_sense_.zero_();

        launch_mfma_xcd_decode_multi(
            (const int*)tok.data_ptr(),
            (int*)cur_tokens_.data_ptr(),
            (int*)out.data_ptr(),
            embed_weight_.data_ptr(),
            (const MFMALayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            lm_head_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_.data_ptr(),
            g_normalized_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_activations_.data_ptr(),
            g_partial_.data_ptr(),
            g_gate_.data_ptr(),
            g_up_.data_ptr(),
            g_mlp_.data_ptr(),
            lm_logits_.data_ptr(),
            (int*)bar_arrive_.data_ptr(),
            (int*)bar_sense_.data_ptr(),
            (const int*)nb_xcd_.data_ptr(),
            batch_, Mpad_, num_layers_, position_, base_cache_len, max_seq_len_,
            attn_scale_, num_steps, stream
        );

        position_ += num_steps;
        return out;
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }
    int batch() const { return batch_; }

private:
    int num_layers_, max_seq_len_, batch_, Mpad_, position_;
    float attn_scale_;

    torch::Tensor embed_weight_, final_norm_weight_, lm_head_weight_;
    torch::Tensor cos_table_, sin_table_, d_layer_weights_;
    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<torch::Tensor> scale_tensors_;
    std::vector<MFMALayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_, g_normalized_, g_residual_, g_activations_, g_partial_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_gate_, g_up_, g_mlp_, lm_logits_, output_tokens_, cur_tokens_;
    torch::Tensor bar_arrive_, bar_sense_, nb_xcd_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelXcdFp8Decoder>(m, "MegakernelXcdFp8Decoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>())
        .def("decode_step", &MegakernelXcdFp8Decoder::decode_step)
        .def("decode_multi", &MegakernelXcdFp8Decoder::decode_multi)
        .def("reset", &MegakernelXcdFp8Decoder::reset)
        .def("position", &MegakernelXcdFp8Decoder::position)
        .def("batch", &MegakernelXcdFp8Decoder::batch);
}
"""

    cls_name = f"MegakernelXcdFp8Decoder_h{hier}"
    cpp_src = cpp_src.replace("MegakernelXcdFp8Decoder", cls_name)

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_flags = cuda_cflags(
        include_dirs=[kernel_dir],
        nvcc_extra=["-arch=sm_86", f"-DMQ_XCD_HIER={hier}", "-DMQ_FP8_WEIGHTS=1"],
    )
    # nvcc_extra is dropped on ROCm; the two defines below must apply on BOTH the
    # cpp (host struct) and cuda (device struct) compiles so their layouts match.
    cuda_flags += [f"-DMQ_XCD_HIER={hier}", "-DMQ_FP8_WEIGHTS=1"]
    mod = load_inline(
        name=f"megakernel_batched_xcd_fp8_h{hier}",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3", "-std=c++17", "-DMQ_FP8_WEIGHTS=1"],
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )
    _xcd_fp8_kernels[hier] = mod
    return mod


def quantize_weights_fp8(weights):
    """bf16 weights dict -> (layer_weights_fp8_flat, scale_flat) for the fp8 kernel.

    Each 2-D projection W[out,in] -> e4m3fnuz codes + per-output-channel f32 scale
    s[out] = amax(|W|,dim=1)/240, such that W ~= fp8_code * s[out]. The 4 norm/
    layernorm tensors in each layer block stay bf16 (not quantized). Returned
    tensors are contiguous and on CUDA; the caller must keep them alive.
    """
    lw = list(weights["layer_weights"])
    nl = len(lw) // 11
    scale_flat = []
    for L in range(nl):
        for j in _PROJ_SLOTS:
            idx = L * 11 + j
            W = lw[idx]
            Wf = W.to(torch.float32)
            amax = Wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)   # [out,1]
            scale = amax / FP8_MAX                                       # [out,1]
            q = (Wf / scale).to(FP8_DTYPE).contiguous()                 # [out,in] fp8
            lw[idx] = q
            scale_flat.append(scale.squeeze(1).to(torch.float32).contiguous())  # [out]
    return lw, scale_flat


class MegakernelXcdGenerator:
    """Fixed-batch generator over the XCD-aware batched MFMA megakernel.

    hier=0 -> grid.sync() everywhere (parity harness: proves sharding + all-reduce).
    hier=1 -> intra-XCD hierarchical barrier (the Stage-2 throughput win).
    """

    def __init__(self, batch, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048,
                 weights=None, hier=1):
        weights = weights or load_qwen3_weights(model_name)
        kernel = _compile_xcd_kernel(hier)

        self.decoder = getattr(kernel, f"MegakernelXcdDecoder_h{hier}")(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            batch,
        )
        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.batch = batch
        self.hier = hier

    def _step(self, tokens_list):
        tok = torch.tensor(tokens_list, dtype=torch.int32, device="cuda")
        out = self.decoder.decode_step(tok)
        return out.tolist()

    def generate_batch(self, prompts, max_new_tokens=100):
        assert len(prompts) == self.batch, f"expected {self.batch} prompts"
        self.decoder.reset()

        ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        L = min(len(x) for x in ids)
        ids = [x[:L] for x in ids]

        for t in range(L - 1):
            self._step([seq[t] for seq in ids])

        generated = [[] for _ in range(self.batch)]
        cur = [seq[L - 1] for seq in ids]
        eos = self.tokenizer.eos_token_id
        done = [False] * self.batch

        for _ in range(max_new_tokens):
            nxt = self._step(cur)
            for b in range(self.batch):
                if not done[b]:
                    if nxt[b] == eos:
                        done[b] = True
                    else:
                        generated[b].append(nxt[b])
            cur = nxt
            if all(done) or self.decoder.position() >= self.max_seq_len - 1:
                break

        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

    def benchmark(self, prompt="The quick brown fox", max_new_tokens=100):
        self.decoder.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for t in ids[:-1]:
            self._step([t] * self.batch)

        cur = [ids[-1]] * self.batch
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(max_new_tokens):
            cur = self._step(cur)
        torch.cuda.synchronize()
        dt = time.time() - t0

        total_tokens = max_new_tokens * self.batch
        return {
            "batch": self.batch,
            "hier": self.hier,
            "new_tokens_per_seq": max_new_tokens,
            "seconds": dt,
            "per_seq_tok_s": max_new_tokens / dt,
            "total_tok_s": total_tokens / dt,
        }

    # ---- In-kernel multi-token path (per-token overhead lever) ----------------
    # The generation phase becomes ONE cooperative launch: decode_multi loops all
    # new tokens on-device, folding the LM head in and feeding argmax back through
    # a device buffer. Greedy => bit-identical tokens to the single-step loop.

    def generate_multi(self, prompts, max_new_tokens=100):
        assert len(prompts) == self.batch, f"expected {self.batch} prompts"
        self.decoder.reset()

        ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        L = min(len(x) for x in ids)
        ids = [x[:L] for x in ids]

        for t in range(L - 1):  # prefill stays host-driven (single-step)
            self._step([seq[t] for seq in ids])

        first = torch.tensor([seq[L - 1] for seq in ids], dtype=torch.int32, device="cuda")
        steps = min(max_new_tokens, self.max_seq_len - self.decoder.position() - 1)
        out = self.decoder.decode_multi(first, steps).cpu().tolist()  # [steps, batch]

        eos = self.tokenizer.eos_token_id
        generated = [[] for _ in range(self.batch)]
        done = [False] * self.batch
        for s in range(steps):
            for b in range(self.batch):
                if done[b]:
                    continue
                tok = out[s][b]
                if tok == eos:
                    done[b] = True
                else:
                    generated[b].append(tok)
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

    def benchmark_multi(self, prompt="The quick brown fox", max_new_tokens=100):
        self.decoder.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for t in ids[:-1]:
            self._step([t] * self.batch)

        first = torch.tensor([ids[-1]] * self.batch, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        out = self.decoder.decode_multi(first, max_new_tokens)
        torch.cuda.synchronize()
        dt = time.time() - t0
        _ = out.cpu()  # realize (excluded from timing)

        total_tokens = max_new_tokens * self.batch
        return {
            "batch": self.batch,
            "hier": self.hier,
            "new_tokens_per_seq": max_new_tokens,
            "seconds": dt,
            "per_seq_tok_s": max_new_tokens / dt,
            "total_tok_s": total_tokens / dt,
        }


class MegakernelXcdFp8Generator(MegakernelXcdGenerator):
    """Weight-only fp8 (W8A16) variant of MegakernelXcdGenerator.

    Same decode/benchmark path; only the decoder differs — projections stored
    e4m3fnuz + per-output-channel scale, dequantized in-register to bf16 for the
    MFMA. Halves projection weight HBM traffic on the bandwidth-bound decode.
    """

    def __init__(self, batch, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048,
                 weights=None, hier=1):
        weights = weights or load_qwen3_weights(model_name)
        kernel = _compile_xcd_fp8_kernel(hier)

        # Quantize projections to fp8 + scales; keep python refs alive alongside
        # the C++ decoder's own tensor refs.
        self._lw_fp8, self._scale_flat = quantize_weights_fp8(weights)

        self.decoder = getattr(kernel, f"MegakernelXcdFp8Decoder_h{hier}")(
            weights["embed_weight"],
            self._lw_fp8,
            self._scale_flat,
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            batch,
        )
        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.batch = batch
        self.hier = hier
