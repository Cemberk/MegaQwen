"""Stage-2 grid-scaling A/B: does the intra-XCD barrier let the grid grow?

The default cooperative grid is capped small (~76 blocks) because a flat
grid.sync() across all 8 XCDs is a cross-die barrier whose cost grows with block
count — filling the device made batch-1 decode collapse. That cap also caps how
many MFMA tiles run concurrently, so it caps throughput once there is enough work
(large B) to use more blocks.

The Stage-2 hierarchical barrier (hier=1) keeps 9 of the 11 per-layer barriers
INTRA-XCD, paying the cross-die sync only twice per layer. If that is the real
lever, then as the grid grows past the cap:
  hier=0 (flat grid.sync everywhere) should degrade — more blocks, costlier sync.
  hier=1 should hold up / improve — extra blocks add GEMM parallelism, and the
          barrier stays cheap.

Grid size is a launch-time parameter fixed per process (MQ_GRID_BLOCKS, read
once). This script sets it from argv, benchmarks hier=0 and hier=1 at a few batch
sizes, and APPENDS rows to a result file (stdout capture under nohup is
unreliable on this host).

Run on a gfx942 (MI300X) host with the Qwen3-0.6B weights cached:
  python csrc/megakernel/bench_xcd_grid.py 304
"""
import os
import sys

# Fix grid + go fully offline BEFORE importing torch / the kernel: the grid size
# is read via getenv at the first cooperative launch, and HF Hub network probes
# otherwise stall the run for minutes on a token-less host.
GRID = sys.argv[1] if len(sys.argv) > 1 else "76"
os.environ["MQ_GRID_BLOCKS"] = GRID
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "csrc/megakernel")

import torch  # noqa: E402
from megakernel_decode import load_qwen3_weights  # noqa: E402
from megakernel_batched_xcd import MegakernelXcdGenerator  # noqa: E402

BATCHES = [8, 32, 64]
NEW = 100
WARMUP = 10
PROMPT = "The quick brown fox jumps over the lazy dog and then"
RESULT = "/tmp/xcd_grid_result.txt"


def emit(line):
    with open(RESULT, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(line, flush=True)


def bench(hier, B, weights):
    gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=hier)
    gen.benchmark(PROMPT, max_new_tokens=WARMUP)
    r = gen.benchmark(PROMPT, max_new_tokens=NEW)
    del gen
    torch.cuda.empty_cache()
    return r["total_tok_s"]


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    emit(f"=== MQ_GRID_BLOCKS={GRID}  (TOTAL tok/s) ===")
    emit(f"{'B':>4} {'h0(flat)':>10} {'h1(xcd)':>10} {'h1/h0':>8}")
    for B in BATCHES:
        try:
            t0 = bench(0, B, weights)
            t1 = bench(1, B, weights)
            emit(f"{B:>4} {t0:>10.1f} {t1:>10.1f} {t1/t0:>7.2f}x")
        except Exception as e:
            emit(f"{B:>4}  ERROR: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
