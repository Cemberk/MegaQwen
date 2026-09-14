# MegaQwen on AMD Instinct (MI300X / ROCm)

Phase-1 port: the single-GPU decode megakernel + the BLAS-free fused-prefill
megakernel run on MI300X (gfx942). Multi-GPU scaling is a later phase.

## Container / environment

Use a ROCm 7.x PyTorch image on an MI300X host (the AMD ODC blog used ROCm 7.2 in
the Primus container; a stock `rocm/pytorch` image works for this single-GPU port):

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  --shm-size 16G \
  -v "$PWD":/workspace -w /workspace \
  rocm/pytorch:latest
```

Inside the container, install Python deps:

```bash
pip install transformers
```

`torch` must be the ROCm build (`python -c "import torch; print(torch.version.hip)"`
should print a version, not `None`).

## Build + verify + benchmark

```bash
bash scripts/rocm/build_and_verify.sh
```

This forces a clean hipcc rebuild of the JIT kernels, runs the parity gate
(`verify_correctness.py`, which hard-asserts greedy-token agreement with
HuggingFace for both the decode and fused-prefill paths), then the benchmark
suite (decode + fused-prefill tok/s vs the HF baseline, labelled with the GPU).

## What was ported (and what to watch)

- **Wavefront 32 -> 64**: `csrc/megakernel/port.cuh` defines `WARP_SIZE`
  per-platform (64 on gfx942); every existing `WARP_SIZE`-relative reduction,
  register-array (`HEAD_DIM/WARP_SIZE`), and lane-pairing recompiles for wave-64.
- **cp.async**: guarded — the AMD path uses a synchronous 128-bit copy fallback
  (correctness first). True async via `__builtin_amdgcn_global_load_lds` is a
  perf follow-up. The default decode kernel (`fused_decode_ldg.cu`) does not use
  cp.async at all.
- **Cooperative grid**: the hardcoded 82-block (3090 SM count) launches now query
  the device (`mq_coop_grid_blocks`) to fill MI300X's 304 CUs. Kernels are
  grid-stride, so co-residency is preserved.
- **Prefill without hipBLAS**: the recommended prefill is the fully-fused
  `fused_prefill_megakernel.cu` (no BLAS). The cuBLAS path (`fused_prefill.cu`)
  also links on ROCm (`-lhipblas` via `ld_flags`) for prompts >64 tokens — its
  bf16 `hipblasGemmEx` numerics should be validated on-device.
- **Perf caveat**: the `__ldg` texture-cache advantage that drove the RTX 3090
  numbers does not transfer to CDNA3, and the "grid.sync-bound ~530 tok/s
  ceiling" must be re-measured on 304 CUs — treat Phase-1 numbers as a
  correctness milestone, not a tuned result.
