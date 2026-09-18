"""Barrier-type A/B: hier=0 (all grid.sync) vs hier=1 (intra-XCD barrier).

usage: python hier_ab.py <B1,B2,...> <N>
Isolates the Stage-2 hierarchical-barrier lever: same kernel, same adaptive grid,
only the 9 per-layer local barriers change from cross-XCD grid.sync (hier=0) to
intra-XCD sense-reversing barrier (hier=1). Prints, per batch, both tok/s and the
speedup. Tokens are also compared to assert hier=0 and hier=1 agree (correctness).
"""
import os
import sys
sys.path.insert(0, "csrc/megakernel")
import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

BATCHES = [int(x) for x in sys.argv[1].split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WARMUP = 8

weights = load_qwen3_weights("Qwen/Qwen3-0.6B")


def run(B, hier):
    gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=hier)
    gen.decoder.reset()
    ids = gen.tokenizer.encode("The quick brown fox", add_special_tokens=True)
    for t in ids[:-1]:
        gen._step([t] * B)
    cur = [ids[-1]] * B
    for _ in range(WARMUP):
        cur = gen._step(cur)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    toks = []
    t0.record()
    for _ in range(N):
        cur = gen._step(cur)
        toks.append(cur[0])
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    tok_s = 1000.0 * B * N / ms
    import gc
    del gen
    gc.collect()
    torch.cuda.empty_cache()
    return tok_s, toks


for B in BATCHES:
    s0, tk0 = run(B, 0)
    s1, tk1 = run(B, 1)
    ok = "MATCH" if tk0 == tk1 else "DIFF"
    print(f"B={B} HIER0_TOK_S={s0:.1f} HIER1_TOK_S={s1:.1f} "
          f"SPEEDUP={s1/s0:.3f}x TOKENS={ok}", flush=True)

torch.cuda.synchronize()
