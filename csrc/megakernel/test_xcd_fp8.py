"""Weight-only fp8 (W8A16) XCD megakernel: parity + throughput A/B (task #28).

Parity: the fp8 kernel must reproduce the bf16 XCD kernel's greedy output on the
confident oracle prompt (the accuracy gate probe already showed fp8 weights are
greedy-identical; this proves the *kernel* dequant+scale path is correct too).

Throughput: fp8 vs bf16 total tok/s at serving batch. Decode is HBM-bound, so
halving projection weight bytes should lift throughput on the weight-streaming
region. Run on a gfx942 (MI300X) host with Qwen3-0.6B weights available:
  python csrc/megakernel/test_xcd_fp8.py
"""
import gc
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator, MegakernelXcdFp8Generator

PROMPT = "Count: 1 2 3 4 5 6 7"
NEW = 40
BENCH_B = 64
BENCH_NEW = 100


def toks(text, tk):
    return tk.encode(text, add_special_tokens=False)


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    tk = weights["tokenizer"]

    print("== bf16 XCD REF (hier=1) ==", flush=True)
    gref = MegakernelXcdGenerator(batch=1, weights=weights, hier=1)
    ref = gref.generate_batch([PROMPT], max_new_tokens=NEW)[0]
    rids = toks(ref, tk)
    print("REF:", repr(ref))
    del gref
    gc.collect(); torch.cuda.empty_cache()

    print("\n== fp8 XCD (hier=1) ==", flush=True)
    gf = MegakernelXcdFp8Generator(batch=1, weights=weights, hier=1)
    out = gf.generate_batch([PROMPT], max_new_tokens=NEW)[0]
    ids = toks(out, tk)
    c = common(rids, ids)
    verdict = "IDENTICAL" if ids == rids else f"DIVERGES@{c}"
    print("FP8:", repr(out))
    print(f"  parity bf16 vs fp8: {c}/{min(len(rids), len(ids))}  {verdict}")
    del gf
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n== throughput A/B  B={BENCH_B}, {BENCH_NEW} new tok ==", flush=True)
    gb = MegakernelXcdGenerator(batch=BENCH_B, weights=weights, hier=1)
    rb = gb.benchmark(max_new_tokens=BENCH_NEW)
    print(f"  bf16: total {rb['total_tok_s']:.0f} tok/s  ({rb['per_seq_tok_s']:.1f}/seq)")
    del gb
    gc.collect(); torch.cuda.empty_cache()

    gfb = MegakernelXcdFp8Generator(batch=BENCH_B, weights=weights, hier=1)
    rf = gfb.benchmark(max_new_tokens=BENCH_NEW)
    print(f"  fp8 : total {rf['total_tok_s']:.0f} tok/s  ({rf['per_seq_tok_s']:.1f}/seq)")

    print("\n================ FP8 XCD SUMMARY ================")
    print(f"  parity : {verdict}")
    print(f"  bf16   : {rb['total_tok_s']:.0f} tok/s")
    print(f"  fp8    : {rf['total_tok_s']:.0f} tok/s")
    print(f"  speedup: {rf['total_tok_s'] / rb['total_tok_s']:.2f}x")


if __name__ == "__main__":
    main()
