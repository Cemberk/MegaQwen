# Phase 2 on-node runbook (MI300X)

Everything here must run **on the MI300X node** (the dev box is Windows/no ROCm).
Two gates block the multi-GPU kernel work: (0) validate Phase 1 + capture the
apples-to-apples baseline, and (1) decide the comm substrate. Do these first and
record the results, then the TP kernel work can proceed against real numbers.

---

## Phase 0 — validate Phase 1 + apples-to-apples baseline

### 0.1 Build + parity + single-GPU bench
```bash
bash scripts/rocm/build_and_verify.sh
python tensor_parallel.py          # confirms torch shape checks pass on-node too
```
Expected: parity gate `[PASS]` (decode + fused-prefill match HF greedy); a printed
tok/s table with the device banner (should read the MI300X). **Record** the
MegaQwen decode + fused-prefill tok/s and the HF tok/s.

### 0.2 Framework stack ON AMD (the only valid apples-to-apples)
The RTX-3090 README table is NOT comparable (different silicon). Build a fresh
**same-hardware** MI300X table by running the ROCm builds of the frameworks on the
same model/prompt/decode-length as `benchmark_suite.py` (Qwen3-0.6B, greedy, 100 tok):

- **vLLM (ROCm)** — has an official ROCm build; benchmark Qwen3-0.6B decode tok/s.
- **SGLang (ROCm)** — ROCm-supported; same measurement.
- **TensorRT-LLM** — no ROCm; **drops out** on AMD (note it, don't fake a number).

Record a table: `HF | MegaQwen-decode | MegaQwen-fused-prefill | vLLM | SGLang`, all
on MI300X. That is where we actually stack.

---

## Phase 1 (of the comm work) — comm-substrate spike (decide before building)

Goal: decide **RCCL GPU-initiated device API** vs **rocSHMEM/MORI** for in-kernel,
GPU-initiated one-sided comm. Record the decision; the `comm.cuh` shim is built
against the winner.

### 1.1 Is there a usable RCCL GPU-initiated *device-side* API?
```bash
# RCCL version
cat /opt/rocm/include/rccl/rccl.h | grep -i version | head
# Look for a device-side / GPU-initiated surface (the NCCL device-API analog):
grep -rIl -i "device" /opt/rocm/include/rccl* 2>/dev/null
grep -rI  -i "ncclDevice\|GpuInitiated\|device_comm\|LSA\|symmetric" /opt/rocm/include/rccl* 2>/dev/null | head
```
Decision: a *host-orchestrated* collective API (`ncclAllReduce`) is NOT what we need
for Tier B — we need device-side put/get/signal or an in-kernel collective callable
from inside the persistent kernel. If RCCL only exposes host collectives in this
ROCm, RCCL is the **Tier A baseline only**, and Tier B uses rocSHMEM.

### 1.2 rocSHMEM / MORI availability (the ODC blog's substrate)
```bash
ls /opt/rocm/include/rocshmem* 2>/dev/null; ls /opt/rocm/lib/librocshmem* 2>/dev/null
# or the source projects:
#   https://github.com/ROCm/rocm-systems  (rocSHMEM / OpenSHMEM)
#   https://github.com/ROCm/mori          (MORI-SHMEM / MORI-IR, NVSHMEM replacement)
python -c "import torch; print('XGMI peer access check below')"
```

### 1.3 Minimal XGMI peer sanity (single node, 2 GPUs)
Confirm direct peer read/write works (intra-node one-sided depends on it):
```bash
python - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "need >=2 GPUs"
a = torch.ones(1<<20, device="cuda:0")
b = torch.empty(1<<20, device="cuda:1")
b.copy_(a)                       # P2P copy over XGMI/Infinity Fabric
torch.cuda.synchronize()
print("P2P copy ok, sum:", b.sum().item())
PY
```

### 1.4 Decision record
Write the outcome into `DEVLOG.md` (a short "Phase 2 comm substrate" note):
- RCCL device API usable? yes/no (+ version)
- rocSHMEM/MORI present? yes/no (+ which)
- **Chosen substrate for Tier B** and why.

---

## After the gates
With the baseline recorded and the substrate chosen, `comm.cuh` (host `comm_init`,
device `allreduce_hidden`, `put_signal`/`wait_signal`) is implemented against the
winner, and the TP kernel work proceeds:
- Tier A: split at the two reduce points (`fused_decode_ldg.cu:496-499`, `:603-605`),
  host-orchestrated RCCL `all_reduce` — validates TP numerics vs HF.
- Tier B: in-kernel GPU-initiated all-reduce at those `grid.sync()` boundaries.
Weight sharding is already implemented and unit-tested in `tensor_parallel.py`.
