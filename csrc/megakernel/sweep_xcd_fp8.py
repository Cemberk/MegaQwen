"""fp8 vs bf16 XCD megakernel: throughput across batch (task #28 characterization).

The weight-only fp8 lever cuts projection weight HBM bytes, so it should help most
in the bandwidth-bound low-batch regime and fade as batch grows and the kernel
turns compute-bound. Sweep total tok/s at several batch sizes to locate the win.
"""
import gc
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator, MegakernelXcdFp8Generator

BATCHES = [1, 4, 16, 64]
NEW = 100


def bench(cls, B, weights):
    g = cls(batch=B, weights=weights, hier=1)
    r = g.benchmark(max_new_tokens=NEW)
    del g
    gc.collect(); torch.cuda.empty_cache()
    return r["total_tok_s"], r["per_seq_tok_s"]


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    print(f"{'B':>4} {'bf16 tot':>10} {'fp8 tot':>10} {'speedup':>8}  (per-seq bf16/fp8)")
    for B in BATCHES:
        bf, bfs = bench(MegakernelXcdGenerator, B, weights)
        f8, f8s = bench(MegakernelXcdFp8Generator, B, weights)
        print(f"{B:>4} {bf:>10.0f} {f8:>10.0f} {f8/bf:>7.2f}x  ({bfs:.1f}/{f8s:.1f})", flush=True)


if __name__ == "__main__":
    main()
