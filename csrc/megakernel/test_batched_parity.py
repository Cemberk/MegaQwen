"""Parity test for the batched MFMA decode megakernel (Stage 1).

Oracle = the LDG baseline (fused_decode_ldg.cu, 221 tok/s, already validated
bit-for-bit vs HuggingFace greedy on confident prompts). We check:
  1. batched B=1  == LDG baseline  (MFMA GEMM path is numerically correct)
  2. batched B=2 (two identical prompts) -> both rows == the B=1 result
     (batching does not cross-contaminate rows / pad handling is clean)

Run inside a ROCm PyTorch container (gfx942):
  TORCH_EXTENSIONS_DIR=/tmp/torch_ext python csrc/megakernel/test_batched_parity.py
"""
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import MegakernelGenerator, load_qwen3_weights
from megakernel_batched import MegakernelBatchedGenerator

PROMPT = "Count: 1 2 3 4 5 6 7"
NEW = 40


def toks(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)


def main():
    print("== loading shared weights ==", flush=True)
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    tok = weights["tokenizer"]

    # --- LDG baseline (oracle) ---
    print("== LDG baseline ==", flush=True)
    ldg = MegakernelGenerator()  # loads its own weights/kernel
    ref_text = ldg.generate(PROMPT, max_new_tokens=NEW)
    ref_ids = toks(ref_text, tok)
    print("LDG  :", repr(ref_text))

    del ldg
    torch.cuda.empty_cache()

    # --- batched B=1 ---
    print("== batched B=1 ==", flush=True)
    gen1 = MegakernelBatchedGenerator(batch=1, weights=weights)
    out1 = gen1.generate_batch([PROMPT], max_new_tokens=NEW)[0]
    ids1 = toks(out1, tok)
    print("B=1  :", repr(out1))

    # --- batched B=2 (identical prompts) ---
    print("== batched B=2 (identical) ==", flush=True)
    gen2 = MegakernelBatchedGenerator(batch=2, weights=weights)
    out2 = gen2.generate_batch([PROMPT, PROMPT], max_new_tokens=NEW)
    ids2a = toks(out2[0], tok)
    ids2b = toks(out2[1], tok)
    print("B=2a :", repr(out2[0]))
    print("B=2b :", repr(out2[1]))

    # --- compare on a common prefix (tie-breaks may diverge deep in) ---
    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    c1 = common(ref_ids, ids1)
    c2a = common(ids1, ids2a)
    c2b = common(ids1, ids2b)
    ab = common(ids2a, ids2b)

    print(f"\nlens: ref={len(ref_ids)} b1={len(ids1)} b2a={len(ids2a)} b2b={len(ids2b)}")
    print(f"match LDG vs B=1     : {c1}/{min(len(ref_ids),len(ids1))}")
    print(f"match B=1 vs B=2[0]  : {c2a}/{min(len(ids1),len(ids2a))}")
    print(f"match B=1 vs B=2[1]  : {c2b}/{min(len(ids1),len(ids2b))}")
    print(f"match B=2[0] vs [1]  : {ab}/{min(len(ids2a),len(ids2b))}")

    ok = (ids1 == ref_ids) and (ids2a == ids1) and (ids2b == ids1)
    # Allow long-prefix agreement (bf16 tie-breaks) as a softer pass signal.
    strong = ok
    soft = c1 >= max(8, len(ref_ids) - 2) and ids2a == ids2b == ids1
    print("\nRESULT:", "STRONG_PASS" if strong else ("SOFT_PASS" if soft else "FAIL"))


if __name__ == "__main__":
    main()
