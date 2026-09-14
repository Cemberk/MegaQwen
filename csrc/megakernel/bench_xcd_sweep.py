"""Stage-2 throughput A/B: software XCD-aware megakernel vs Stage-1 batched.

Three variants, one process, same weights, same workload:
  stage1  -> flat-grid batched MFMA megakernel (megakernel_batched.py).
  h0      -> XCD sharding + all-reduce, but grid.sync() everywhere (isolates the
             sharding cost from the barrier).
  h1      -> XCD sharding + intra-XCD hierarchical barrier (the Stage-2 win).

The Stage-1 flat grid.sync() is a cross-die barrier whose cost grows with the
number of cooperative blocks, so per-seq tok/s collapses as B rises. The Stage-2
claim is that replacing the ~9 local phase boundaries per layer with intra-XCD
barriers keeps per-seq throughput from collapsing, lifting TOTAL tok/s at the
batch sizes that matter for serving. h0 vs h1 isolates how much of any change is
the barrier vs the sharding.

Run on a gfx942 (MI300X) host with the Qwen3-0.6B weights available:
  python csrc/megakernel/bench_xcd_sweep.py
"""
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched import MegakernelBatchedGenerator
from megakernel_batched_xcd import MegakernelXcdGenerator

BATCHES = [1, 2, 4, 8, 16, 32, 64]
NEW = 100
WARMUP = 10
PROMPT = "The quick brown fox jumps over the lazy dog and then"


def bench_one(make_gen, B, weights):
    """Build a fixed-batch generator, warm it up, return (per_seq, total) tok/s."""
    gen = make_gen(B, weights)
    gen.benchmark(PROMPT, max_new_tokens=WARMUP)
    r = gen.benchmark(PROMPT, max_new_tokens=NEW)
    del gen
    torch.cuda.empty_cache()
    return r["per_seq_tok_s"], r["total_tok_s"]


VARIANTS = [
    ("stage1", lambda B, w: MegakernelBatchedGenerator(batch=B, weights=w)),
    ("xcd-h0", lambda B, w: MegakernelXcdGenerator(batch=B, weights=w, hier=0)),
    ("xcd-h1", lambda B, w: MegakernelXcdGenerator(batch=B, weights=w, hier=1)),
]


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")

    # results[name][B] = (per_seq, total)
    results = {name: {} for name, _ in VARIANTS}
    for name, make in VARIANTS:
        for B in BATCHES:
            try:
                results[name][B] = bench_one(make, B, weights)
            except Exception as e:
                print(f"[{name} B={B}] ERROR: {type(e).__name__}: {e}", flush=True)
                results[name][B] = None
                torch.cuda.empty_cache()

    # Per-seq tok/s (does the flat-grid collapse show, and does h1 avoid it?)
    print("\n================ per-seq tok/s ================")
    print(f"{'B':>4} {'stage1':>10} {'xcd-h0':>10} {'xcd-h1':>10} {'h1/stage1':>10}")
    print("-" * 48)
    for B in BATCHES:
        s1 = results["stage1"].get(B)
        h0 = results["xcd-h0"].get(B)
        h1 = results["xcd-h1"].get(B)
        s1p = s1[0] if s1 else float("nan")
        h0p = h0[0] if h0 else float("nan")
        h1p = h1[0] if h1 else float("nan")
        ratio = (h1p / s1p) if (s1 and h1 and s1p) else float("nan")
        print(f"{B:>4} {s1p:>10.1f} {h0p:>10.1f} {h1p:>10.1f} {ratio:>9.2f}x")

    # TOTAL tok/s (aggregate serving throughput)
    print("\n================ TOTAL tok/s ================")
    print(f"{'B':>4} {'stage1':>10} {'xcd-h0':>10} {'xcd-h1':>10} {'h1/stage1':>10}")
    print("-" * 48)
    for B in BATCHES:
        s1 = results["stage1"].get(B)
        h0 = results["xcd-h0"].get(B)
        h1 = results["xcd-h1"].get(B)
        s1t = s1[1] if s1 else float("nan")
        h0t = h0[1] if h0 else float("nan")
        h1t = h1[1] if h1 else float("nan")
        ratio = (h1t / s1t) if (s1 and h1 and s1t) else float("nan")
        print(f"{B:>4} {s1t:>10.1f} {h0t:>10.1f} {h1t:>10.1f} {ratio:>9.2f}x")

    def peak(name):
        pts = [(B, v[1]) for B, v in results[name].items() if v]
        return max(pts, key=lambda x: x[1]) if pts else (None, float("nan"))

    print("-" * 48)
    for name, _ in VARIANTS:
        B, t = peak(name)
        print(f"peak TOTAL {name:>7}: {t:>8.1f} tok/s @ B={B}")


if __name__ == "__main__":
    main()
