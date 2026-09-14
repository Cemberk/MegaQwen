#!/usr/bin/env bash
# Build (JIT), parity-check, and benchmark MegaQwen on an AMD Instinct MI300X node.
#
# Run this on the ROCm host (not the Windows dev box). It forces a clean hipcc
# rebuild of the JIT extensions, runs the real parity gate against HuggingFace,
# then the benchmark suite. Exits non-zero if parity fails.
#
# Usage:
#   bash scripts/rocm/build_and_verify.sh
#   PYTORCH_ROCM_ARCH=gfx942 bash scripts/rocm/build_and_verify.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# MI300X = gfx942. Set so torch's hipify build targets the right arch.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"

echo "==== Environment ===="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("cuda(available):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
assert getattr(torch.version, "hip", None), "This is not a ROCm PyTorch build — aborting."
assert torch.cuda.is_available(), "No ROCm GPU visible — check --device=/dev/kfd,/dev/dri."
PY

# Force a clean rebuild of the JIT extensions so hipcc recompiles the ported
# kernels (torch caches by source hash in ~/.cache/torch_extensions).
echo "==== Clearing torch_extensions cache ===="
rm -rf "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"/*megakernel* 2>/dev/null || true
rm -rf "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"/*ldg_kernel* 2>/dev/null || true

echo "==== Parity gate (megakernel vs HuggingFace) ===="
python verify_correctness.py --tokens 20 --min-match 8

echo "==== Benchmark suite ===="
python benchmark_suite.py

echo "==== Done ===="
