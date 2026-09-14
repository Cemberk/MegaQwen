"""
Batched + MFMA megakernel decode for Qwen3-0.6B (MI300X / gfx942).

Wraps csrc/megakernel/fused_decode_mfma.cu — the Stage-1 batched megakernel that
turns each projection into an MFMA bf16 GEMM (weight tile loaded once for all B
rows). Kept separate from megakernel_decode.py so the 221 tok/s LDG baseline
stays untouched as the A/B reference.

Stage-1 scope: LOCKSTEP fixed-batch decode — a decoder instance is built for a
fixed batch size B; all B sequences share one `position` (the standard
fixed-length throughput benchmark). Variable/continuous batching is a later
refinement (the KV cache is already laid out per-sequence to allow it).
"""

import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from build_flags import cuda_cflags  # noqa: E402

# Reuse constants + the HF weight loader from the baseline module.
from megakernel_decode import (  # noqa: E402
    load_qwen3_weights,
    HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_Q_HEADS, NUM_KV_HEADS,
    HEAD_DIM, Q_SIZE, KV_SIZE, NUM_LAYERS, VOCAB_SIZE,
)

_batched_kernel = None


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_batched_kernel():
    global _batched_kernel
    if _batched_kernel is not None:
        return _batched_kernel

    # The MFMA builtin (__builtin_amdgcn_mfma_*) only exists on CDNA
    # (gfx90a/942/950). Without this, torch's load_inline builds for every
    # ROCm arch (incl. RDNA gfx10xx/11xx) and fails. MI300X is gfx942.
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

    cuda_src = _get_cuda_source("fused_decode_mfma.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Must match MFMALayerWeights in fused_decode_mfma.cu (11 bf16 pointers).
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

extern "C" void launch_mfma_decode(
    const int* input_tokens, int* output_tokens,
    const void* embed_weight, const MFMALayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache,
    void* hidden, void* g_normalized, void* g_residual,
    void* g_q, void* g_k, void* g_v,
    void* g_attn_out, void* g_activations, void* g_gate, void* g_up, void* g_mlp,
    void* lm_logits,
    int B, int Mpad, int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream
);

static inline int round_up16(int x) { return ((x + 15) / 16) * 16; }

class MegakernelBatchedDecoder {
public:
    MegakernelBatchedDecoder(
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

        // Per-sequence KV cache: [batch, layers, kv_heads, max_seq, head_dim].
        k_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);
        v_cache_ = torch::zeros({batch_, num_layers, NUM_KV_HEADS, max_seq_len, HEAD_DIM}, bf16);

        // Padded-M work buffers. Allocated ZEROED so pad rows [B, Mpad) start at 0;
        // per-row phases only touch rows < B and GEMM pad tiles restore 0, so pad
        // rows stay clean for the life of the (fixed-batch) decoder.
        hidden_       = torch::zeros({Mpad_, HIDDEN_SIZE}, bf16);
        g_normalized_ = torch::zeros({Mpad_, HIDDEN_SIZE}, bf16);
        g_residual_   = torch::zeros({Mpad_, HIDDEN_SIZE}, f32);
        g_q_          = torch::zeros({Mpad_, Q_SIZE}, f32);
        g_k_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_v_          = torch::zeros({Mpad_, KV_SIZE}, f32);
        g_attn_out_   = torch::zeros({Mpad_, Q_SIZE}, bf16);
        g_activations_= torch::zeros({Mpad_, HIDDEN_SIZE}, f32);
        g_gate_       = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_up_         = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, f32);
        g_mlp_        = torch::zeros({Mpad_, INTERMEDIATE_SIZE}, bf16);
        lm_logits_    = torch::zeros({Mpad_, VOCAB_SIZE}, f32);
        output_tokens_= torch::zeros({batch_}, i32);

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

    // tokens: int32 CUDA tensor [batch]. Returns int32 CUDA tensor [batch].
    torch::Tensor decode_step(torch::Tensor tokens) {
        TORCH_CHECK(tokens.dtype() == torch::kInt32, "tokens must be int32");
        TORCH_CHECK(tokens.numel() == batch_, "tokens must have batch elements");
        auto tok = tokens.to(torch::kCUDA).contiguous();

        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_mfma_decode(
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
            g_gate_.data_ptr(),
            g_up_.data_ptr(),
            g_mlp_.data_ptr(),
            lm_logits_.data_ptr(),
            batch_, Mpad_, num_layers_, position_, cache_len, max_seq_len_,
            attn_scale_, stream
        );

        position_++;
        return output_tokens_.clone();
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
    torch::Tensor hidden_, g_normalized_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_, g_activations_;
    torch::Tensor g_gate_, g_up_, g_mlp_, lm_logits_, output_tokens_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelBatchedDecoder>(m, "MegakernelBatchedDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>())
        .def("decode_step", &MegakernelBatchedDecoder::decode_step)
        .def("reset", &MegakernelBatchedDecoder::reset)
        .def("position", &MegakernelBatchedDecoder::position)
        .def("batch", &MegakernelBatchedDecoder::batch);
}
"""

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    _batched_kernel = load_inline(
        name="megakernel_batched",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=cuda_cflags(include_dirs=[kernel_dir], nvcc_extra=["-arch=sm_86"]),
        verbose=False,
    )
    return _batched_kernel


class MegakernelBatchedGenerator:
    """Fixed-batch generator over the batched MFMA megakernel.

    All B sequences decode in lockstep from the same position. Used for the
    batch-size throughput sweep (tok/s vs B) against the LDG baseline and vLLM.
    """

    def __init__(self, batch, model_name="Qwen/Qwen3-0.6B", max_seq_len=2048, weights=None):
        weights = weights or load_qwen3_weights(model_name)
        kernel = _compile_batched_kernel()

        self.decoder = kernel.MegakernelBatchedDecoder(
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

    def _step(self, tokens_list):
        tok = torch.tensor(tokens_list, dtype=torch.int32, device="cuda")
        out = self.decoder.decode_step(tok)
        return out.tolist()

    def generate_batch(self, prompts, max_new_tokens=100):
        """Greedy-decode a list of exactly `batch` prompts in lockstep.

        Prompts are left-unpadded and assumed equal effective length for the
        lockstep benchmark; each is fed token-by-token, then generation runs.
        Returns list[str] of length batch.
        """
        assert len(prompts) == self.batch, f"expected {self.batch} prompts"
        self.decoder.reset()

        ids = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        L = min(len(x) for x in ids)
        ids = [x[:L] for x in ids]  # lockstep: common prefill length

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
        """Total-throughput microbenchmark: `batch` identical sequences."""
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
            "new_tokens_per_seq": max_new_tokens,
            "seconds": dt,
            "per_seq_tok_s": max_new_tokens / dt,
            "total_tok_s": total_tokens / dt,
        }
