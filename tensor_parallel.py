"""Tensor-parallel weight sharding for MegaQwen (Qwen3 dense).

Megatron-style TP: hidden activations are full/replicated at layer boundaries
(after the all-reduce); only the projection weights are sharded within a layer.

  - column-parallel (slice output dim / dim0): q_proj, k_proj, v_proj, gate_proj, up_proj
  - row-parallel    (slice input  dim / dim1): o_proj, down_proj
  - vocab-parallel  (slice dim0):              lm_head
  - replicated (unchanged):                    embeddings, all norms, RoPE tables

Rank r's kernel is compiled/run with LOCAL dims (Q_SIZE // tp, etc.) and all-reduces
the O-proj and down-proj partial sums (see the Phase 2 plan for the exact insertion
points). This module ONLY produces correctly-sliced tensors — no GPU, no comm — so
its per-rank index math is unit-testable on CPU (torch is imported lazily, only for
the actual tensor slicing).
"""

from dataclasses import dataclass

# Qwen3-0.6B dims — mirror csrc/megakernel/config.cuh
HIDDEN_SIZE = 1024
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_Q_HEADS * HEAD_DIM          # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM        # 1024
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 151936

# Weights that are sharded vs replicated under TP.
COLUMN_PARALLEL = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")
ROW_PARALLEL = ("o_proj.weight", "down_proj.weight")


@dataclass(frozen=True)
class TPConfig:
    rank: int
    world: int

    def __post_init__(self):
        if not (0 <= self.rank < self.world):
            raise ValueError(f"rank {self.rank} out of range for tp_size {self.world}")


def even_split(total: int, world: int, rank: int):
    """Contiguous [start, end) slice of `total` for `rank`. Requires even divisibility."""
    if total % world != 0:
        raise ValueError(f"{total} not divisible by tp_size {world}")
    per = total // world
    return rank * per, (rank + 1) * per


def head_split(num_heads: int, head_dim: int, world: int, rank: int):
    """Element [start, end) for a per-head-sharded projection (heads split across ranks)."""
    if num_heads % world != 0:
        raise ValueError(f"{num_heads} heads not divisible by tp_size {world}")
    heads_per_rank = num_heads // world
    return rank * heads_per_rank * head_dim, (rank + 1) * heads_per_rank * head_dim


def vocab_split(world: int, rank: int):
    """[start, end) vocab slice for lm_head. Ceil split so it works even if VOCAB doesn't
    divide evenly (Qwen3 vocab 151936 does divide by 1/2/4/8, but keep it general)."""
    per = (VOCAB_SIZE + world - 1) // world
    start = rank * per
    end = min(start + per, VOCAB_SIZE)
    return start, end


def local_dims(world: int) -> dict:
    """Per-rank LOCAL sizes the rank's kernel must be built with (as -D overrides of config.cuh)."""
    for name, n in (("NUM_Q_HEADS", NUM_Q_HEADS), ("NUM_KV_HEADS", NUM_KV_HEADS),
                    ("INTERMEDIATE_SIZE", INTERMEDIATE_SIZE)):
        if n % world != 0:
            raise ValueError(f"{name}={n} not divisible by tp_size {world}")
    return {
        "NUM_Q_HEADS": NUM_Q_HEADS // world,
        "NUM_KV_HEADS": NUM_KV_HEADS // world,
        "Q_SIZE": Q_SIZE // world,
        "KV_SIZE": KV_SIZE // world,
        "INTERMEDIATE_SIZE": INTERMEDIATE_SIZE // world,
        "HIDDEN_SIZE": HIDDEN_SIZE,   # unchanged: hidden is full/replicated at boundaries
        "HEAD_DIM": HEAD_DIM,         # unchanged
    }


def shard_layer(w: dict, tp: TPConfig) -> dict:
    """Slice one layer's HF weights for tp.rank. `w` maps short names -> torch tensors
    (e.g. 'q_proj.weight'). Replicated entries are returned unchanged. tp.world == 1 is a
    no-op copy. Requires torch only for `.contiguous()` on the sliced views."""
    if tp.world == 1:
        return dict(w)
    qs, qe = head_split(NUM_Q_HEADS, HEAD_DIM, tp.world, tp.rank)
    ks, ke = head_split(NUM_KV_HEADS, HEAD_DIM, tp.world, tp.rank)
    is_, ie = even_split(INTERMEDIATE_SIZE, tp.world, tp.rank)

    out = dict(w)
    # column-parallel (slice dim0 = output rows)
    out["q_proj.weight"] = w["q_proj.weight"][qs:qe, :].contiguous()
    out["k_proj.weight"] = w["k_proj.weight"][ks:ke, :].contiguous()
    out["v_proj.weight"] = w["v_proj.weight"][ks:ke, :].contiguous()
    out["gate_proj.weight"] = w["gate_proj.weight"][is_:ie, :].contiguous()
    out["up_proj.weight"] = w["up_proj.weight"][is_:ie, :].contiguous()
    # row-parallel (slice dim1 = input cols); all-reduce reassembles the output
    out["o_proj.weight"] = w["o_proj.weight"][:, qs:qe].contiguous()
    out["down_proj.weight"] = w["down_proj.weight"][:, is_:ie].contiguous()
    # everything else (input_layernorm, post_attn_layernorm, q_norm, k_norm) is replicated
    return out


def shard_lm_head(lm_head_weight, tp: TPConfig):
    """Vocab-parallel slice (dim0). Returns (sliced_weight, (start, end))."""
    if tp.world == 1:
        return lm_head_weight, (0, VOCAB_SIZE)
    start, end = vocab_split(tp.world, tp.rank)
    return lm_head_weight[start:end, :].contiguous(), (start, end)


# ------------------------------------------------------------------------------
# Self-test: index math only (no torch), plus torch shape checks if torch present.
# Run: python tensor_parallel.py
# ------------------------------------------------------------------------------
def _selftest_index_math():
    for world in (1, 2, 4, 8):
        # column/head splits must tile the full range with no gaps/overlaps
        q_bounds = [head_split(NUM_Q_HEADS, HEAD_DIM, world, r) for r in range(world)]
        assert q_bounds[0][0] == 0 and q_bounds[-1][1] == Q_SIZE, (world, q_bounds)
        assert all(q_bounds[r][1] == q_bounds[r + 1][0] for r in range(world - 1)), q_bounds

        kv_bounds = [head_split(NUM_KV_HEADS, HEAD_DIM, world, r) for r in range(world)]
        assert kv_bounds[-1][1] == KV_SIZE, (world, kv_bounds)

        int_bounds = [even_split(INTERMEDIATE_SIZE, world, r) for r in range(world)]
        assert int_bounds[-1][1] == INTERMEDIATE_SIZE, (world, int_bounds)

        voc_bounds = [vocab_split(world, r) for r in range(world)]
        assert voc_bounds[0][0] == 0 and voc_bounds[-1][1] == VOCAB_SIZE, (world, voc_bounds)
        covered = sum(e - s for s, e in voc_bounds)
        assert covered == VOCAB_SIZE, (world, covered)

        ld = local_dims(world)
        assert ld["Q_SIZE"] * world == Q_SIZE
        assert ld["INTERMEDIATE_SIZE"] * world == INTERMEDIATE_SIZE
    print("[ok] index math: head/even/vocab splits tile exactly for tp in {1,2,4,8}")


def _selftest_shapes():
    try:
        import torch
    except ImportError:
        print("[skip] torch not installed; shape check skipped (index math already verified)")
        return
    for world in (2, 4, 8):
        tp = TPConfig(rank=1, world=world)
        w = {
            "q_proj.weight": torch.zeros(Q_SIZE, HIDDEN_SIZE),
            "k_proj.weight": torch.zeros(KV_SIZE, HIDDEN_SIZE),
            "v_proj.weight": torch.zeros(KV_SIZE, HIDDEN_SIZE),
            "o_proj.weight": torch.zeros(HIDDEN_SIZE, Q_SIZE),
            "gate_proj.weight": torch.zeros(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "up_proj.weight": torch.zeros(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "down_proj.weight": torch.zeros(HIDDEN_SIZE, INTERMEDIATE_SIZE),
            "input_layernorm.weight": torch.zeros(HIDDEN_SIZE),
        }
        s = shard_layer(w, tp)
        assert tuple(s["q_proj.weight"].shape) == (Q_SIZE // world, HIDDEN_SIZE)
        assert tuple(s["o_proj.weight"].shape) == (HIDDEN_SIZE, Q_SIZE // world)
        assert tuple(s["gate_proj.weight"].shape) == (INTERMEDIATE_SIZE // world, HIDDEN_SIZE)
        assert tuple(s["down_proj.weight"].shape) == (HIDDEN_SIZE, INTERMEDIATE_SIZE // world)
        assert tuple(s["input_layernorm.weight"].shape) == (HIDDEN_SIZE,)  # replicated
        lm, (st, en) = shard_lm_head(torch.zeros(VOCAB_SIZE, HIDDEN_SIZE), tp)
        assert tuple(lm.shape) == (en - st, HIDDEN_SIZE)
    print("[ok] torch shapes: sharded projections have expected per-rank dims")


if __name__ == "__main__":
    _selftest_index_math()
    _selftest_shapes()
