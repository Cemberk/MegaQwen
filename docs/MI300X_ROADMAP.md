# MegaQwen on AMD MI300X — Status & Roadmap

Living record of the CUDA→ROCm port and the MI300X optimization effort.
The local checkout is the source of truth; build/run happens on an MI300X node
in a self-created ROCm PyTorch container (gfx942) (see `scripts/rocm/`).

## Latest results (2026-09-14) — batch-adaptive cooperative grid

The XCD-aware decode megakernel now sizes its cooperative grid **per batch**, which
recovers **+15–27% throughput at batch ≤ 32** with **bit-identical greedy output**
(the grid change is launch-config only; per-thread math is unchanged — parity PASS,
40/40 tokens vs the full-device grid).

Validated TOTAL tok/s (adaptive grid auto-selects the measured optimum per batch):

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|-------|---|---|---|---|----|----|----|
| tok/s | 192 | 383 | 763 | 1455 | 2806 | 4603 | 7230 |
| gain over device-fill | +27% | +27% | +26% | +22% | +15% | +5% | 0% |

**Apples-to-apples vs vLLM (same MI300X, same 100-tok greedy workload).** vLLM
v0.27.1 ROCm TOTAL tok/s were measured under identical conditions (task #14; vLLM
unchanged since, so the comparison is valid). The batch-adaptive grid roughly
*doubles* MegaQwen's share of vLLM versus the earlier Stage-1 MFMA path:

| B | MegaQwen (adaptive) | vLLM v0.27.1 | MegaQwen ÷ vLLM | (was, Stage-1 MFMA) |
|---|---|---|---|---|
| 1  | 192  | 396   | 0.48× | 0.30× |
| 4  | 763  | 1993  | 0.38× | 0.23× |
| 8  | 1455 | 3443  | 0.42× | 0.25× |
| 16 | 2806 | 6856  | 0.41× | 0.22× |
| 32 | 4603 | 12616 | 0.36× | 0.19× |
| 64 | 7230 | 23461 | 0.31× | 0.14× |

Honest read: **vLLM's continuous batching + graph capture still leads at every
batch size**, and the gap widens with B (barrier cost grows with the cooperative
grid). But the single-persistent-kernel design is now within ~2–3× at serving
batches, up from ~4–7×, entirely from launch-config topology awareness — no change
to the per-thread math.

**The hierarchical-barrier lever is measured, and it is a *substitute* for the
adaptive grid, not additive (task #27, closed).** There are two independent ways to
attack the cross-XCD barrier cost: (A) launch *fewer* cooperative blocks so the flat
`grid.sync` has fewer cross-die participants (the adaptive grid), or (B) keep the full
grid but replace the 9 per-layer local barriers with the intra-XCD sense-reversing
barrier (`MQ_XCD_HIER=1`, S0c), leaving only the 2 true cross-XCD reductions/layer as
`grid.sync`. An A/B at B=1 isolates them (`hier_ab.py`, bit-identical tokens both ways):

| grid | hier=0 (all `grid.sync`) | hier=1 (intra-XCD) | hier speedup |
|---|---|---|---|
| 304 (full device) | 77.0 | 132.8 | **1.73×** |
| 76 (adaptive)     | 190.4 | 190.6 | 1.00× (wash) |

So the intra-XCD barrier is a genuine **1.7× lever on a full-device grid** — but the
adaptive grid alone (190) already beats hier-on-full-grid (133), and stacking hier on
top of the adaptive grid is a wash (1.00×, and likewise 0.996–1.004× at B=8/32). Both
levers cash out the *same* barrier-serialization cost, so they don't compose. The
adaptive grid is kept as the default; the hierarchical barrier stays a
validated-correct compile switch (`MQ_XCD_HIER`, default on) for regimes forced to a
full grid, and no barrier-*count* reduction is warranted under the current per-batch
grid policy. Remaining gap to vLLM is therefore scheduling/batching (continuous
batching + graph capture), not barrier structure.

**Why it works — the megakernel is barrier-bound at low batch, not bandwidth-bound.**
Profiling the single-step kernel showed the limiter is `grid.sync()` / barrier
serialization across the 8 XCD chiplets (~140 barriers/token), *not* HBM bandwidth,
occupancy, or host overhead:
- A trace attributes ~94% of wall-clock to the decode kernel, ~2% to host.
- An occupancy A/B *falsifies* the occupancy hypothesis: the kernel can co-reside up
  to 4 blocks/CU, yet adding co-resident blocks makes it **monotonically slower**
  (B=1: 1/CU 151 → 2/CU 107 → 4/CU 66 tok/s).
- A grid-size sweep shows a **batch-dependent optimum**: fewer cooperative blocks =
  fewer barrier participants = a cheaper cross-die barrier. High batch (GEMM-bound)
  still wants a full device grid.

Policy (portable, as fractions of the device-fill base): **B≤4 → base/4, B≤32 →
base/2, else base**, snapped to a multiple of the 8 XCDs so the intra-XCD barrier's
arrival target matches the launched grid. This also explains why the earlier fp8
weight-only and async-LDS experiments were ~null — they cut memory/compute traffic,
which is not what bounds this kernel at serving batch sizes.

Reproduce: `python csrc/megakernel/parity.py 1 40` (greedy parity) and
`python csrc/megakernel/grid_sweep.py 1,2,4,8,16,32,64 20` (throughput sweep) inside
a ROCm PyTorch container on an MI300X.

## Status (2026-08-25)

- **Phase 1 — CUDA→ROCm port: DONE, validated on real MI300X (gfx942, ROCm 7.14).**
  Compiles under hipcc; numerically correct (24/24 exact greedy match vs HuggingFace
  on a confident prompt; other divergences are bf16 tie-breaks).
- **First optimization: DONE.** Naive port was 0.6× HF due to cooperative-grid
  over-subscription; capping the grid → **3.9× HF**.
- **Phase 3 Stage 1 — batched + MFMA decode: DONE, parity-green + benchmarked.**
  New `fused_decode_mfma.cu` + `megakernel_batched.py`; all projections are now
  16×16×16 bf16 MFMA GEMM tiles (`mfma.cuh`). Bit-parity: batched B=1 == LDG
  baseline == B=2 rows, 40/40 tokens on the counting oracle. Throughput sweep
  (below) crosses vLLM's batch-1 number between B=4 and B=8; **peak 3256 tok/s @
  B=64 = 14.7× the LDG baseline**. Built/run in a self-created ROCm container.

### Benchmark — MI300X, Qwen3-0.6B, 100-tok greedy decode, FAIR B-vs-B (2026-08-25)
Both engines: same prompt, exactly 100 output tokens/seq (MegaQwen fixed-100;
vLLM `ignore_eos + min_tokens=100`), greedy, one MI300X GPU (GPU 0), TOTAL tok/s.
vLLM = v0.27.1 ROCm in a self-created `--rm` container (V1 engine, CUDA graphs,
end-to-end incl. the ~11-tok prefill).

| B | MegaQwen MFMA TOTAL | vLLM v0.27.1 TOTAL | MegaQwen ÷ vLLM |
|---|---|---|---|
| 1  | 117  | 396   | 0.30× |
| 4  | 465  | 1993  | 0.23× |
| 8  | 870  | 3443  | 0.25× |
| 16 | 1541 | 6856  | 0.22× |
| 32 | 2398 | 12616 | 0.19× |
| 64 | 3256 | **23461** | **0.14×** |

**Honest read (this corrects the earlier "6.09× vLLM" framing):** that number
compared MegaQwen's B=64 TOTAL against vLLM's *batch-1* figure — apples-to-oranges.
On a fair B-vs-B basis **vLLM is ~4–7× faster at every batch size, and the gap
widens with B** (0.30× at B=1 → 0.14× at B=64). MegaQwen per-seq collapses
117→51 across B=1→64 (sublinear); vLLM holds ~366–536 per-seq. This is exactly
the S0c prediction: the flat cross-XCD `grid.sync` (~252/token) is a hard
~3.1 ms/token ≈ 320 tok/s/seq ceiling that vLLM sidesteps with continuous
batching + graph capture. **The fair table does not undercut the plan — it is the
strongest motivation for Stage 2 (XCD-aware hierarchical barrier + weight
sharding), which attacks precisely this ceiling.** SGLang row still TODO.

Nuance retained: at B=1 the MFMA path (117) is *slower* than the tuned LDG GEMV
(221) — M=1 padded to 16 wastes 15/16 of each tile; LDG stays the batch-1
latency kernel, MFMA is the (still-uncompetitive) throughput path.

### Benchmark — MI300X, Qwen3-0.6B, batch-1, 100-tok greedy decode
| Backend | tok/s | vs HF |
|---|---|---|
| PyTorch HF (eager) | 57 | 1.0× |
| MegaQwen (ported + grid-tuned) | 221 | 3.9× |
| vLLM (ROCm) | 535 | 9.4× |
| SGLang (ROCm) | TODO | — |

The vLLM 535 here is an earlier decode-throughput reading (task #14); the fair
end-to-end sweep above measures vLLM B=1 at 396 and per-seq peaking ~536 at B=2 —
consistent once prefill/scheduling overhead is folded into the B=1 denominator.

Cross-hardware caveat: the RTX-3090 numbers in `README.md` are NOT comparable
(different silicon; the megakernel's 3090 win over vLLM does not hold on MI300X).

## Core insight driving the roadmap
Batch-1 decode is **HBM-bandwidth-bound** (re-streams all weights per token,
arithmetic intensity ~1, matrix cores idle). The megakernel's "minimal kernel
movement" advantage only pays off once **compute-bound**, which requires
**batching** (GEMV→GEMM, weights loaded once per B tokens, MFMA usable).
MI300X is **8 XCD chiplets (~38 CUs each, private L2) over Infinity Fabric = a
NUMA hierarchy** — a flat `grid.sync()` across all XCDs is the dominant cost
(912 blocks → 35 tok/s vs 76 → 221).

## Phase 3 — MI300X single-GPU optimization (CURRENT)

### Stage 0 — de-risk spikes (standalone hipcc kernels, fast loop)
- **S0a — MFMA bf16 primitive. [DONE]** `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`
  (bf16×4 in, f32×4 acc) is **bit-exact** vs a reference GEMM on gfx942. Layout: lane%16 =
  A-row/B-col/D-col, lane/16 = K-block(inputs)/M-block(output), 4 elems/lane. Naive
  single-chain 23.9 TFLOP/s (real tiling later). `experiments/mi300x/mfma_gemm.hip`.
- **S0b — read physical XCD/XCC id. [DONE]** In-kernel `s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)`
  returns 0–7 on gfx942 (HW_ID1/HW_ID2 are NOT valid there). **Block→XCD placement is
  deterministic round-robin: `blockIdx.x % 8 == XCC_ID`** (38/XCD at 304 blocks, 304/XCD at
  2432) → software XCD control needs no CPX. `experiments/mi300x/xcd_probe.hip`.
- **S0c — hierarchical intra-XCD barrier. [DONE 2026-08-25]** Sense-reversing per-XCD
  barrier (per-XCD atomic counter, `nb_xcd` = blocks-per-XCD under blockIdx%8==XCC).
  `experiments/mi300x/xcd_barrier.hip`. **Intra-XCD is 6.5× cheaper than flat grid.sync
  at the 76-block cap (1.9 vs 12.4 µs/barrier); gap widens to 14× at 608** because
  grid.sync scales ~linearly with block count (cross-fabric) while intra-XCD stays flat.
  Implication: ~252 barriers/token × 12.4 µs ≈ 3.1 ms/token in barriers alone (~320 tok/s
  ceiling) — demoting all but the true cross-XCD reductions is the Stage-2 win.
- **Standalone hipcc binaries need** `LD_LIBRARY_PATH=.../_rocm_sdk_devel/lib`.

### Stage 1 — batched decode: GEMV → MFMA GEMM (the core lever) [DONE 2026-08-25]
New `csrc/megakernel/fused_decode_mfma.cu` (keep `fused_decode_ldg.cu` as A/B baseline).
Delivered: `megakernel_batched.py` (`MegakernelBatchedDecoder` fixed-batch C++ +
`MegakernelBatchedGenerator`), `mfma.cuh` tile w/ fused-residual store. Build must
pin `PYTORCH_ROCM_ARCH=gfx942` (MFMA builtin is CDNA-only; load_inline's default
all-arch list fails on gfx10xx). Parity + sweep above. Next lever = Stage 2 (the
sublinear knee past B=8 is the flat cross-XCD `grid.sync`).
- `[B, dim]` bf16 activation buffers (f32 accumulate); projections become
  `[B,in] @ [in,out]^T` MFMA GEMM tiles (weight tile loaded once for all B rows).
- Per-sequence KV cache `[B, layers, kv_heads, max_seq, head_dim]` + `position_[B]`;
  attention per-(b,q-head), GQA ratio 2 preserved.
- Batched API `decode_step(Tensor[B]) → Tensor[B]` + `MegakernelBatchedGenerator`.
  Prefill folds in (prefill = batched multi-token).
- **Success metric:** total throughput vs LDG-baseline + vLLM across B ∈ {1..64};
  curve should bend toward/over vLLM as B grows.

### Stage 2 — software XCD-aware execution (SPX, single model)
Each block reads its XCD id (S0b), self-assigns to the weight shard cached in that
XCD's L2 (+ NPS memory mode); hierarchical barrier (S0c) replaces the ~140×/token
flat `grid.sync` — full cross-XCD barrier only where a reduction spans XCDs
(O-proj + down-proj all-reduce, final norm). Same cut points as Phase-2 TP.

- **Hierarchical barrier: DONE + measured (task #27, 2026-09-18).** Wired into the
  batched kernel as the `MQ_XCD_HIER` compile switch (per layer: 2 cross-XCD
  `grid.sync` + 9 intra-XCD `mq_xcd_bar`); `hier=0/1` compile to separate modules
  (`megakernel_batched_xcd_h{0,1}`) so an in-process A/B is exact. Result (see "Latest
  results" table above): **1.73× at B=1 on a full-device grid, but a wash at the
  adaptive grid** — the adaptive grid already extracts the barrier win, so the two are
  substitutes. Bit-identical tokens both ways. Conclusion: adaptive grid is the
  default lever; the hier barrier is retained as a validated compile option, and
  barrier-count reduction is **not** pursued because it is non-additive under the
  current grid policy. The remaining Stage-2 item (XCD-local weight *sharding* in each
  XCD's L2) is orthogonal to the barrier and still open.

### Stage 3 — compose with CPX (throughput)
CPX = 8 XCD partitions; run one batched-MFMA instance per partition (data-parallel
replicas) with software XCD-awareness within. Compare SPX-XCD-aware (one big model)
vs CPX 8×replicas for aggregate serving throughput.

### Stage 4 — async LDS + arch tuning
Replace the ROCm synchronous cp.async fallback (`port.cuh mq_async_copy16`) with real
`__builtin_amdgcn_global_load_lds` double-buffering; tune MFMA tile / wavefronts /
LDS / occupancy for gfx942.

## Phase 2 — multi-GPU scaling (LATER)
Both **tensor-parallel** and **expert-parallel (MoE)** axes; single-node
(8×MI300X / XGMI) first, then multi-node (RDMA). Comm substrate = RCCL
GPU-initiated device API, fallback rocSHMEM/MORI; classic RCCL collectives = baseline.
TP weight-sharding is already implemented + unit-tested (`tensor_parallel.py`); TP
all-reduce cut points are `fused_decode_ldg.cu:496-499` (O-proj) and `:603-605`
(down-proj). References: UniEP (MoE megakernel), Perseus (multi-node GPU-initiated
put-with-signal), AMD ODC/FSDP blog.

## Key gotchas (see also DEVLOG.md)
- torch `load_inline` hipify rewrites the inline `.cu` string but NOT `#include`d
  disk headers → keep platform specifics in `port.cuh`.
- HIP warp-sync builtins need a 64-bit mask (`WARP_FULL_MASK`).
- HIP `__ldg` lacks vector/bf16 overloads → `LDG()` template (plain load on ROCm).
- Cooperative grid must be capped (~76), not device-filled.
- `__ldg` texture-cache advantage does not exist on CDNA3.

## Operational quickref
On an MI300X (gfx942) node, in your own ROCm PyTorch container, build/verify/bench:
`bash scripts/rocm/build_and_verify.sh`. Create your own `--rm` container from a
published image rather than reusing a shared one. vLLM image:
`vllm/vllm-openai-rocm:nightly-...`; SGLang image: `rocm/sgl-dev`.
