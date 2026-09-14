"""Steady-state single-step decode driver for rocprofv3 tracing (task #29 follow-up).

Builds the XCD hier=1 decoder, prefills, warms up, then runs a marked STEADY
region of N single-step decode_step calls. rocprofv3 --runtime-trace captures
each kernel dispatch; the parser subtracts summed GPU kernel time from the
wall-clock STEADY region to split per-token time into device vs host-overhead.

usage: python prof_decode.py <batch> <nsteps>
"""
import sys
sys.path.insert(0, "csrc/megakernel")
import torch
from megakernel_decode import load_qwen3_weights
from megakernel_batched_xcd import MegakernelXcdGenerator

B = int(sys.argv[1])
N = int(sys.argv[2])
WARMUP = 8

weights = load_qwen3_weights("Qwen/Qwen3-0.6B")
gen = MegakernelXcdGenerator(batch=B, weights=weights, hier=1)

# prefill
gen.decoder.reset()
ids = gen.tokenizer.encode("The quick brown fox", add_special_tokens=True)
for t in ids[:-1]:
    gen._step([t] * B)
cur = [ids[-1]] * B

# warmup (compile/caches settle)
for _ in range(WARMUP):
    cur = gen._step(cur)
torch.cuda.synchronize()

# marked steady region
torch.cuda.nvtx.range_push("STEADY")
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record()
for _ in range(N):
    cur = gen._step(cur)
t1.record()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

ms = t0.elapsed_time(t1)
print(f"STEADY_WALL_MS={ms:.4f} B={B} N={N} PER_TOK_US={1000.0*ms/N:.2f} TOK_S={1000.0*B*N/ms:.1f}",
      flush=True)

# Deterministic teardown: free the cooperative-kernel extension buffers while HSA
# is still up, to dodge the ROCm/torch destructor-ordering SIGSEGV at interpreter
# shutdown (which otherwise makes rocprofv3 counter mode abort before flushing).
import gc
del gen
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()

