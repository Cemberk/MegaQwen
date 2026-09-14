"""In-kernel multi-token decode: parity vs single-step + throughput A/B (task #29).

The single-step path pays per-token host/launch overhead (~97% of B=1 wall clock:
tensor build + .tolist() D2H sync + barrier memsets + cooperative relaunch + 2 LM
head launches). decode_multi folds the LM head into the cooperative kernel and
loops all new tokens on-device, feeding each step's greedy argmax back through a
device buffer — ONE launch for the whole generation phase.

Greedy argmax is deterministic, so the multi-step tokens must be BIT-IDENTICAL to
the single-step loop. We assert that first (correctness), then measure the win.
"""
import gc
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

PROMPTS = ["The quick brown fox", "Count: 1 2 3 4 5 6 7", "Once upon a time"]
BATCHES = [1, 4, 16, 64]
NEW = 100
PARITY_N = 48


def raw_single(gen, prompt, n):
    gen.decoder.reset()
    ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
    for t in ids[:-1]:
        gen._step([t] * gen.batch)
    cur = [ids[-1]] * gen.batch
    seq = []
    for _ in range(n):
        cur = gen._step(cur)
        seq.append(list(cur))
    return seq  # [n][batch]


def raw_multi(gen, prompt, n):
    gen.decoder.reset()
    ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
    for t in ids[:-1]:
        gen._step([t] * gen.batch)
    first = torch.tensor([ids[-1]] * gen.batch, dtype=torch.int32, device="cuda")
    return gen.decoder.decode_multi(first, n).cpu().tolist()  # [n][batch]


def parity(weights):
    print("=== PARITY: decode_multi vs single-step decode_step ===", flush=True)
    ok = True
    for B in (1, 4):
        gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=1)
        for p in PROMPTS:
            s = raw_single(gen, p, PARITY_N)
            m = raw_multi(gen, p, PARITY_N)
            match = (s == m)
            ok = ok and match
            # locate first divergence if any
            div = "-"
            if not match:
                for i in range(PARITY_N):
                    if s[i] != m[i]:
                        div = f"step {i}: single={s[i][0]} multi={m[i][0]}"
                        break
            print(f"  B={B:>2} {p[:22]:<22} {'MATCH' if match else 'DIFF'}  {div}", flush=True)
        del gen
        gc.collect(); torch.cuda.empty_cache()
    print(f"PARITY_RESULT={'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def throughput(weights):
    print("\n=== THROUGHPUT: single-step vs in-kernel multi (total tok/s) ===", flush=True)
    print(f"{'B':>4} {'single tot':>11} {'multi tot':>11} {'speedup':>8}  (per-seq s/m)", flush=True)
    for B in BATCHES:
        gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=1)
        rs = gen.benchmark(max_new_tokens=NEW)
        rm = gen.benchmark_multi(max_new_tokens=NEW)
        print(f"{B:>4} {rs['total_tok_s']:>11.0f} {rm['total_tok_s']:>11.0f} "
              f"{rm['total_tok_s'] / rs['total_tok_s']:>7.2f}x  "
              f"({rs['per_seq_tok_s']:.1f}/{rm['per_seq_tok_s']:.1f})", flush=True)
        del gen
        gc.collect(); torch.cuda.empty_cache()


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    ok = parity(weights)
    throughput(weights)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
