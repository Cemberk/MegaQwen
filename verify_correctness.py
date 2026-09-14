"""Parity gate: the megakernel's greedy decode must match HuggingFace Qwen3-0.6B.

This is a REAL assertion-based check. (The previous version only printed a note
and always exited 0, so it could not catch a broken port.) It validates two
paths against the HF reference and exits non-zero on mismatch:

  1. decode path      - MegakernelGenerator (prompt run through the decode kernel)
  2. fused-prefill    - MegakernelFusedPrefillGenerator (one-shot prefill megakernel)

Runs on CUDA or ROCm/HIP (torch device is 'cuda' on both). Greedy decoding is
deterministic, so a correct kernel must reproduce HF's leading tokens; bf16
fusion may diverge only in the tail, hence the `--min-match` threshold rather
than requiring all tokens to agree.
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "megakernel"))

MODEL = "Qwen/Qwen3-0.6B"


def backend_name() -> str:
    return "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"


def hf_greedy_ids(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    return out[0, ids.shape[1]:].tolist()


def mega_decode_ids(gen, tok, prompt, n):
    """Drive MegakernelGenerator's decoder token-by-token (its decode_step is argmax/greedy)."""
    gen.decoder.reset()
    input_ids = tok.encode(prompt, add_special_tokens=True)
    for t in input_ids[:-1]:          # prefill simulation via the decode kernel
        gen.decoder.decode_step(t)
    ids, cur = [], input_ids[-1]
    for _ in range(n):
        cur = int(gen.decoder.decode_step(cur))
        ids.append(cur)
    return ids


def mega_fused_prefill_ids(gen, tok, prompt, n):
    """Drive the fused-prefill decoder: one-shot prefill_step, then greedy decode."""
    dec = gen.decoder
    if hasattr(dec, "reset"):
        dec.reset()
    input_ids = tok.encode(prompt, add_special_tokens=True)
    first = int(dec.prefill_step(torch.tensor(input_ids, dtype=torch.int32)))
    ids, cur = [first], first
    for _ in range(n - 1):
        cur = int(dec.decode_step(cur))
        ids.append(cur)
    return ids


def leading_match(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def check(label, hf_ids, mega_ids, need, tok):
    ml = leading_match(hf_ids, mega_ids)
    print(f"\n[{label}]")
    print(f"  HF   ids: {hf_ids}")
    print(f"  Mega ids: {mega_ids}")
    print(f"  HF   text: {tok.decode(hf_ids)!r}")
    print(f"  Mega text: {tok.decode(mega_ids)!r}")
    print(f"  Leading greedy-token agreement: {ml}/{len(hf_ids)} (need >= {need})")
    ok = True
    if not mega_ids or hf_ids[0] != mega_ids[0]:
        print(f"  [FAIL] first token differs (HF={hf_ids[0]}, Mega={mega_ids[0] if mega_ids else None})")
        ok = False
    if ml < need:
        print(f"  [FAIL] only {ml} leading tokens agree (< {need})")
        ok = False
    if ok:
        print("  [PASS]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Megakernel vs HuggingFace parity gate")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--min-match", type=int, default=8, help="required leading greedy-token agreement")
    ap.add_argument("--skip-fused", action="store_true", help="skip the fused-prefill path check")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] No CUDA/ROCm device visible; parity gate needs a GPU.")
        return 0

    print(f"Backend: {backend_name()} | device: {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    hf_ids = hf_greedy_ids(hf, tok, args.prompt, args.tokens)
    need = min(args.min_match, args.tokens)

    from megakernel_decode import MegakernelGenerator
    decode_ok = check("decode", hf_ids, mega_decode_ids(MegakernelGenerator(), tok, args.prompt, args.tokens), need, tok)

    fused_ok = True
    if not args.skip_fused:
        prompt_len = len(tok.encode(args.prompt, add_special_tokens=True))
        if prompt_len > 64:
            print(f"\n[fused-prefill] skipped: prompt {prompt_len} tokens > 64 (fused prefill cap). "
                  "Use cuBLAS prefill (MegakernelPrefillGenerator) for long prompts.")
        else:
            try:
                from megakernel_decode import MegakernelFusedPrefillGenerator
                fused_ok = check(
                    "fused-prefill", hf_ids,
                    mega_fused_prefill_ids(MegakernelFusedPrefillGenerator(), tok, args.prompt, args.tokens),
                    need, tok,
                )
            except Exception as e:  # binding/build issue on this path shouldn't be silent
                print(f"\n[fused-prefill] [FAIL] could not run fused-prefill path: {e}")
                fused_ok = False

    if decode_ok and fused_ok:
        print("\n[PASS] Megakernel matches HuggingFace greedy decode (decode + fused-prefill).")
        return 0
    print("\n[FAIL] Parity gate failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
