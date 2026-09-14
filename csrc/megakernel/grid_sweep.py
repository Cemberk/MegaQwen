"""Optimal-grid(B) sweep: fixed MQ_GRID_BLOCKS (read once/process), loop batches.

usage: MQ_GRID_BLOCKS=<g> python grid_sweep.py <B1,B2,...> <N>
Loads weights once, then for each batch builds an XCD hier=1 generator, warms up,
and times a steady region of N single-step decodes. Prints one line per batch:
  GRID=<g> B=<b> PER_TOK_US=<..> TOK_S=<..>
"""
import os
import sys
sys.path.insert(0, "csrc/megakernel")
import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

BATCHES = [int(x) for x in sys.argv[1].split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
WARMUP = 8
G = os.environ.get("MQ_GRID_BLOCKS", "auto")

weights = load_qwen3_weights("Qwen/Qwen3-0.6B")

for B in BATCHES:
    gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=1)
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
    t0.record()
    for _ in range(N):
        cur = gen._step(cur)
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    print(f"GRID={G} B={B} PER_TOK_US={1000.0*ms/N:.2f} TOK_S={1000.0*B*N/ms:.1f}", flush=True)
    import gc
    del gen
    gc.collect()
    torch.cuda.empty_cache()

# deterministic-ish teardown (benign SIGSEGV at HSA shutdown is harmless here)
torch.cuda.synchronize()
