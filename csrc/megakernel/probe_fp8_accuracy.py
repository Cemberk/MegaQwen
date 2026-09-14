"""fp8 weight-quantization ACCURACY de-risk (Stage-4, task #28).

The decode projections are HBM-bandwidth-bound at serving batch (see DEVLOG /
project memory): the one lever that moves a bandwidth-bound kernel is cutting
weight bytes. Before doing any fp8-MFMA kernel work, answer the cheaper, bigger
risk first: does fp8 weight quantization preserve Qwen3-0.6B's *greedy* output?

Method: fake-quantize the projection weights (q/k/v/o/gate/up/down) bf16 ->
e4m3fnuz -> bf16 (MI300's native fp8 is FNUZ, not OCP e4m3) and run the EXISTING,
validated bf16 XCD megakernel (hier=1). No new CUDA. Compare greedy tokens on the
confident oracle prompt against the un-quantized bf16 reference. If greedy output
survives fp8 weights, the accuracy risk is cleared and the fp8-MFMA kernel work is
worth doing; if it breaks even here, we need finer granularity (per-channel/group)
or int8 -- which this probe also A/Bs.

Run on a gfx942 (MI300X) host with Qwen3-0.6B weights available:
  python csrc/megakernel/probe_fp8_accuracy.py
"""
import sys
sys.path.insert(0, "csrc/megakernel")

import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

PROMPT = "Count: 1 2 3 4 5 6 7"
NEW = 40
PER_LAYER = 11
PROJ = {1, 2, 3, 6, 8, 9, 10}   # q,k,v,o,gate,up,down within each 11-tensor block

assert hasattr(torch, "float8_e4m3fnuz"), "torch lacks float8_e4m3fnuz on this build"
FP8 = torch.float8_e4m3fnuz
FP8_MAX = 240.0   # e4m3fnuz max representable magnitude


def toks(text, tk):
    return tk.encode(text, add_special_tokens=False)


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def q_per_channel(W):
    Wf = W.to(torch.float32)
    amax = Wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / FP8_MAX
    return ((Wf / scale).to(FP8).to(torch.float32) * scale).to(torch.bfloat16)


def q_per_tensor(W):
    Wf = W.to(torch.float32)
    amax = Wf.abs().amax().clamp(min=1e-8)
    scale = amax / FP8_MAX
    return ((Wf / scale).to(FP8).to(torch.float32) * scale).to(torch.bfloat16)


def build_quant(weights, qfn, include_lm_head=False):
    w = dict(weights)
    lw = list(weights["layer_weights"])
    nl = len(lw) // PER_LAYER
    errs = []
    for L in range(nl):
        for j in PROJ:
            idx = L * PER_LAYER + j
            W = lw[idx]
            Wq = qfn(W)
            denom = W.to(torch.float32).abs().amax().clamp(min=1e-8)
            errs.append(((W.to(torch.float32) - Wq.to(torch.float32)).abs().amax() / denom).item())
            lw[idx] = Wq.contiguous()
    w["layer_weights"] = lw
    if include_lm_head:
        w["lm_head_weight"] = qfn(weights["lm_head_weight"]).contiguous()
    print(f"  quant maxrel over {len(errs)} weights: max {max(errs):.4f}  mean {sum(errs)/len(errs):.4f}")
    return w


def gen(weights):
    g = MegakernelXcdGenerator(batch=1, weights=weights, hier=1)
    out = g.generate_batch([PROMPT], max_new_tokens=NEW)[0]
    del g
    torch.cuda.empty_cache()
    return out


def main():
    weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
    tk = weights["tokenizer"]

    print("== REF (bf16, unquantized) ==", flush=True)
    ref = gen(weights)
    rids = toks(ref, tk)
    print("REF :", repr(ref))

    trials = [
        ("fp8 per-channel", q_per_channel, False),
        ("fp8 per-tensor", q_per_tensor, False),
        ("fp8 per-channel +lm_head", q_per_channel, True),
    ]
    results = []
    for name, qfn, lmh in trials:
        print(f"\n== {name} ==", flush=True)
        wq = build_quant(weights, qfn, lmh)
        out = gen(wq)
        ids = toks(out, tk)
        c = common(rids, ids)
        verdict = "IDENTICAL" if ids == rids else f"DIVERGES@{c}"
        print(f"{name}:", repr(out))
        print(f"  match REF vs {name}: {c}/{min(len(rids), len(ids))}  {verdict}")
        results.append((name, c, len(rids), ids == rids))
        del wq, out
        torch.cuda.empty_cache()

    print("\n================ FP8 ACCURACY SUMMARY ================")
    for name, c, n, ident in results:
        print(f"  {name:28s}: {c}/{n}  {'PASS(identical greedy)' if ident else 'DIVERGES'}")


if __name__ == "__main__":
    main()
