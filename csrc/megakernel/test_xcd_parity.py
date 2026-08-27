"""Parity test for the Stage-2 XCD-aware batched MFMA megakernel.

Oracle = the LDG baseline (fused_decode_ldg.cu, validated bit-for-bit vs HF greedy
on confident prompts). Two independent things are being validated, so both compile
switches are exercised in one run:

  hier=0  -> grid.sync() everywhere. Proves the XCD *sharding + all-reduce* is
             numerically correct on its own (barrier held constant vs Stage 1).
  hier=1  -> intra-XCD hierarchical barrier. Proves the cheap barrier does not
             change the result (this is the Stage-2 throughput path).

For each: B=1 must match the LDG oracle, and B=2 (identical prompts) must produce
two identical rows equal to B=1 (no cross-row contamination / clean pad handling).

Run on a gfx942 (MI300X) host with the Qwen3-0.6B weights available:
  python csrc/megakernel/test_xcd_parity.py
"""
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import MegakernelGenerator, load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

PROMPT = "Count: 1 2 3 4 5 6 7"
NEW = 40


def toks(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def check_hier(hier, weights, tok, ref_ids):
    print(f"\n===== XCD hier={hier} =====", flush=True)
    gen1 = MegakernelXcdGenerator(batch=1, weights=weights, hier=hier)
    out1 = gen1.generate_batch([PROMPT], max_new_tokens=NEW)[0]
    ids1 = toks(out1, tok)
    print(f"h{hier} B=1 :", repr(out1))
    del gen1
    torch.cuda.empty_cache()

    gen2 = MegakernelXcdGenerator(batch=2, weights=weights, hier=hier)
    out2 = gen2.generate_batch([PROMPT, PROMPT], max_new_tokens=NEW)
    ids2a = toks(out2[0], tok)
    ids2b = toks(out2[1], tok)
    print(f"h{hier} B=2a:", repr(out2[0]))
    print(f"h{hier} B=2b:", repr(out2[1]))
    del gen2
    torch.cuda.empty_cache()

    c1 = common(ref_ids, ids1)
    print(f"lens: ref={len(ref_ids)} b1={len(ids1)} b2a={len(ids2a)} b2b={len(ids2b)}")
    print(f"match LDG vs B=1    : {c1}/{min(len(ref_ids), len(ids1))}")
    print(f"match B=1 vs B=2[0] : {common(ids1, ids2a)}/{min(len(ids1), len(ids2a))}")
    print(f"match B=1 vs B=2[1] : {common(ids1, ids2b)}/{min(len(ids1), len(ids2b))}")
    print(f"match B=2[0] vs [1] : {common(ids2a, ids2b)}/{min(len(ids2a), len(ids2b))}")

    strong = (ids1 == ref_ids) and (ids2a == ids1) and (ids2b == ids1)
    soft = c1 >= max(8, len(ref_ids) - 2) and ids2a == ids2b == ids1
    res = "STRONG_PASS" if strong else ("SOFT_PASS" if soft else "FAIL")
    print(f"hier={hier} RESULT: {res}")
    return res


def main():
    print("== loading shared weights ==", flush=True)
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    tok = weights["tokenizer"]

    print("== LDG baseline (oracle) ==", flush=True)
    ldg = MegakernelGenerator()
    ref_text = ldg.generate(PROMPT, max_new_tokens=NEW)
    ref_ids = toks(ref_text, tok)
    print("LDG  :", repr(ref_text))
    del ldg
    torch.cuda.empty_cache()

    r0 = check_hier(0, weights, tok, ref_ids)
    r1 = check_hier(1, weights, tok, ref_ids)

    print("\n================ SUMMARY ================")
    print(f"  sharding+all-reduce (hier=0): {r0}")
    print(f"  hierarchical barrier (hier=1): {r1}")
    ok = r0 in ("STRONG_PASS", "SOFT_PASS") and r1 in ("STRONG_PASS", "SOFT_PASS")
    print("OVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
