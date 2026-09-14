"""Greedy-decode parity probe for the adaptive cooperative grid.

The batch-adaptive grid changes ONLY the launch configuration (block count +
nb_xcd[] arrival targets), never the per-thread math. So greedy tokens MUST be
bit-identical to the device-fill baseline. Force the grid via MQ_GRID_BLOCKS to
A/B: run once at 304 (device-fill) and once adaptive (unset) and diff TOKENS=.

usage: [MQ_GRID_BLOCKS=<g>] python parity.py <B> <NGEN>
"""
import os
import sys
sys.path.insert(0, "csrc/megakernel")
import torch  # noqa: F401
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 40

weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=1)
gen.decoder.reset()
ids = gen.tokenizer.encode("The quick brown fox", add_special_tokens=True)
for t in ids[:-1]:
    gen._step([t] * B)
cur = [ids[-1]] * B
out = []
for _ in range(NGEN):
    cur = gen._step(cur)
    out.append(int(cur[0]))
print(f"GRID={os.environ.get('MQ_GRID_BLOCKS','auto')} B={B} TOKENS=" + " ".join(map(str, out)), flush=True)
