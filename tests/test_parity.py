"""On-device parity test: megakernel greedy decode must match HuggingFace.

Skips automatically when no CUDA/ROCm GPU is present (e.g. CI on CPU), so it is
safe to include in the default suite. On an MI300X node it builds the kernels
and asserts the parity gate passes.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="parity gate needs a CUDA/ROCm GPU",
)


def test_megakernel_matches_huggingface(monkeypatch):
    import verify_correctness as vc

    # Short run keeps the on-device test quick while still exercising decode +
    # fused-prefill against the HF reference.
    monkeypatch.setattr(sys, "argv", ["verify_correctness", "--tokens", "12", "--min-match", "5"])
    assert vc.main() == 0
