"""Control for the Stage-2 grid-scaling result: does plain Stage-1 (flat batched
MFMA, no XCD sharding) also scale when its cooperative grid cap is lifted?

The XCD kernel went 3325 -> 6196 tok/s at B=64 as the grid grew 76 -> 304. That
win could be (a) XCD sharding making a big grid usable, or (b) just more MFMA
tiles in flight — which plain Stage-1 would get too. This runs the Stage-1 kernel
across the same grid sizes so the two curves can be compared directly.

Grid size is fixed per process via MQ_GRID_BLOCKS (read once at first launch), set
here from argv before importing the kernel. Results are appended to a file.

  python csrc/megakernel/bench_stage1_grid.py 304
"""
import os
import sys

GRID = sys.argv[1] if len(sys.argv) > 1 else "76"
os.environ["MQ_GRID_BLOCKS"] = GRID
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "csrc/megakernel")

import torch  # noqa: E402
from megakernel_decode import load_qwen3_weights  # noqa: E402
from megakernel_batched import MegakernelBatchedGenerator  # noqa: E402

BATCHES = [8, 32, 64]
NEW = 100
WARMUP = 10
PROMPT = "The quick brown fox jumps over the lazy dog and then"
RESULT = "/tmp/stage1_grid_result.txt"


def emit(line):
    with open(RESULT, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(line, flush=True)


def bench(B, weights):
    gen = MegakernelBatchedGenerator(batch=B, weights=weights)
    gen.benchmark(PROMPT, max_new_tokens=WARMUP)
    r = gen.benchmark(PROMPT, max_new_tokens=NEW)
    del gen
    torch.cuda.empty_cache()
    return r["total_tok_s"]


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    emit(f"=== stage1  MQ_GRID_BLOCKS={GRID}  (TOTAL tok/s) ===")
    emit(f"{'B':>4} {'stage1':>10}")
    for B in BATCHES:
        try:
            emit(f"{B:>4} {bench(B, weights):>10.1f}")
        except Exception as e:
            emit(f"{B:>4}  ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
