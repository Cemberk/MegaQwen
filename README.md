# MegaQwen

Custom CUDA megakernel for Qwen3-0.6B inference achieving **530 tok/s decode** on RTX 3090 (3.9x faster than HuggingFace).

## Performance

| Backend | Decode (tok/s) | Speedup |
|---------|---------------|---------|
| **Megakernel** | **531** | **3.9x** |
| TensorRT-LLM | 355 | 2.6x |
| vLLM | 107 | 0.8x |
| SGLang | 107 | 0.8x |
| HuggingFace | 136 | 1.0x |

**Note**: Decode throughput depends on context length. At position 1: 525 tok/s, at position 200: 422 tok/s. See [experiments/RESULTS.md](experiments/RESULTS.md) for full benchmarks.

## AMD MI300X (ROCm) — this fork

This fork ports the megakernel from CUDA/RTX-3090 to **AMD MI300X (gfx942, ROCm 7.x)**
and rebuilds the decode path to be **batched, MFMA-based, and XCD-topology-aware**.
All numbers below are Qwen3-0.6B, greedy decode, one MI300X GPU. Full methodology and
the CUDA→ROCm port notes are in [docs/MI300X_ROADMAP.md](docs/MI300X_ROADMAP.md).

**Single-stream (batch 1, 100-tok decode):**

| Backend | Decode (tok/s) | Speedup vs HF |
|---------|---------------|---------------|
| **MegaQwen (this fork)** | **221** | **3.9×** |
| vLLM (ROCm) | 535 | 9.4× |
| SGLang (ROCm) | 523 | 9.2× |
| HuggingFace (eager) | 57 | 1.0× |

**Batched total throughput** — the XCD-aware megakernel with a **batch-adaptive
cooperative grid** (validated, parity-identical to a full-device grid):

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|-------|---|---|---|---|----|----|----|
| TOTAL tok/s | 192 | 383 | 763 | 1455 | 2806 | 4603 | 7230 |

The batch-adaptive grid recovers **+15–27% at batch ≤ 32** over a naive
device-filling grid, with **bit-identical greedy output**. The reason: at low batch
this cooperative megakernel is **barrier-bound** — its ~140 `grid.sync()` barriers per
token, not HBM bandwidth or occupancy, set the ceiling — so launching *fewer*
cooperative blocks makes each cross-die barrier cheaper. The grid size is therefore
chosen per batch (`base/4` for B≤4, `base/2` for B≤32, full device otherwise).

**Honest read:** production engines with continuous batching (vLLM ROCm) still lead
at large batch on MI300X; this fork's value is the single-persistent-kernel design,
competitive single-stream latency, and a measured, reproducible analysis of *what*
actually bounds a cooperative megakernel on a chiplet GPU. See the roadmap for the
full fair batch-vs-batch comparison.

## Fair Comparison (Devil's Advocate)

Credit where it's due: **TensorRT-LLM, vLLM, SGLang, and other frameworks are excellently optimized for production workloads** with dynamic shapes, variable batch sizes, and long contexts. This megakernel exploits several advantages they intentionally don't:

1. **Static shapes**: All dimensions (hidden size, head count, MLP width) are compile-time constants. Production frameworks must handle arbitrary model architectures at runtime.

2. **Short context bias**: The benchmarks favor position 1-100 where KV cache overhead is minimal. At longer contexts, TensorRT-LLM's consistent 355 tok/s beats the megakernel's degradation to 158 tok/s.

3. **Single model, single GPU**: No tensor parallelism, no continuous batching, no dynamic memory allocation. Real serving systems need all of these.

4. **Learning exercise**: This project was built to understand GPU optimization, not to replace production inference engines.

The speedup is real, but it comes from exploiting a narrow regime (batch=1, short context, static shapes) where the **texture cache (`__ldg()`) provides massive benefits** by keeping weights in the read-only cache path while L1/L2 handles activations. Production frameworks can't make these assumptions.

**TL;DR**: Use TensorRT-LLM or vLLM for production. Use this to learn how GPUs actually work.

## What is a Megakernel?

A megakernel fuses an entire transformer block into a single CUDA kernel launch, eliminating kernel launch overhead and intermediate memory traffic. This implementation:

- Fuses RMSNorm, QKV projection, RoPE, attention, O projection, and MLP into one kernel
- Uses `__ldg()` for cached weight reads via texture cache
- Employs cooperative groups for grid-wide synchronization
- Implements online softmax for memory-efficient attention

## Requirements

- NVIDIA GPU with compute capability 8.6+ (RTX 3090, A100, etc.)
- CUDA 11.8+
- Python 3.10+

## Installation

```bash
git clone https://github.com/Infatoshi/MegaQwen.git
cd MegaQwen

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers triton
```

## Usage

### Interactive Chat
```bash
python chat.py
```

### Run Benchmarks
```bash
python benchmark_suite.py
```

### Verify Correctness
```bash
python verify_correctness.py
```

## Key Findings

After exhaustive optimization, we discovered the kernel is **latency-bound by synchronization, not memory bandwidth**:

- **5% memory bandwidth utilization** (47 GB/s effective vs 936 GB/s peak)
- **140+ `grid.sync()` calls** per token at ~0.7us each = ~100us sync overhead
- **~530 tok/s is the architectural ceiling** for batch=1 bf16 cooperative megakernels on RTX 3090

### What Worked

| Optimization | Impact |
|--------------|--------|
| Block divergence + L2 prefetch | **+2x** |
| 128-bit vectorized loads | +3.5% |

### What Didn't Work

| Optimization | Impact | Why |
|--------------|--------|-----|
| Warp producer/consumer split | 0% | Reduces compute parallelism |
| Shared memory caching | 0% | L1/L2 already effective |
| cp.async double-buffering | +1% | Can't overlap enough compute |

See [DEVLOG.md](DEVLOG.md) for the complete optimization journey.

## Documentation

- **[Development Log](DEVLOG.md)** - Complete optimization journey and learnings
- [Benchmark Results](experiments/RESULTS.md) - Full benchmark data
- [Memory Analysis](docs/MEMORY_ANALYSIS.md) - Bandwidth and SASS analysis
- [Architecture](docs/ARCHITECTURE.md) - Kernel architecture details
- [Specification](SPEC.md) - Technical specification

## License

MIT
