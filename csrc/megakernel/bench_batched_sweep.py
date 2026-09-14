"""Stage-1 throughput sweep: batched MFMA megakernel vs LDG baseline.

Reports per-seq and TOTAL tok/s across batch sizes. The megakernel thesis is
that batching turns the HBM-bound GEMV path into a compute-bound MFMA GEMM, so
TOTAL throughput should climb with B and bend toward / past vLLM (535 tok/s on
this node) even though per-seq latency may dip.

Run inside a ROCm PyTorch container (gfx942):
  TORCH_EXTENSIONS_DIR=/tmp/torch_ext python csrc/megakernel/bench_batched_sweep.py
"""
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched import MegakernelBatchedGenerator

BATCHES = [1, 2, 4, 8, 16, 32, 64]
NEW = 100
PROMPT = "The quick brown fox jumps over the lazy dog and then"


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    print(f"\n{'B':>4} {'per-seq tok/s':>14} {'TOTAL tok/s':>12} {'vs vLLM535':>10}")
    print("-" * 44)
    rows = []
    for B in BATCHES:
        try:
            gen = MegakernelBatchedGenerator(batch=B, weights=weights)
            # warmup
            gen.benchmark(PROMPT, max_new_tokens=10)
            r = gen.benchmark(PROMPT, max_new_tokens=NEW)
            rows.append((B, r["per_seq_tok_s"], r["total_tok_s"]))
            print(f"{B:>4} {r['per_seq_tok_s']:>14.1f} {r['total_tok_s']:>12.1f} "
                  f"{r['total_tok_s']/535:>9.2f}x")
            del gen
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{B:>4}  ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    if rows:
        best = max(rows, key=lambda x: x[2])
        print("-" * 44)
        print(f"peak TOTAL: {best[2]:.1f} tok/s @ B={best[0]}  "
              f"(LDG baseline 221, vLLM 535)")


if __name__ == "__main__":
    main()
