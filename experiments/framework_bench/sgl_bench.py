"""SGLang (ROCm) offline throughput — apples-to-apples with the vLLM #14 row.

Mirrors the vLLM offline methodology exactly: same prompt, greedy (temperature 0),
exactly 100 output tokens/seq (ignore_eos + max_new_tokens=100), one MI300X GPU,
TOTAL tok/s = B * 100 / wall over a batch of B identical prompts, after one warmup
generate (folds in CUDA-graph capture / prefill, same as the vLLM measurement).

The `if __name__ == "__main__"` guard is REQUIRED: SGLang launches its scheduler
with multiprocessing 'spawn', which re-imports this module in the child; without the
guard the child would re-create the Engine and the scheduler dies during init.

Run inside a self-created rocm/sgl-dev --rm container on GPU 0:
    python experiments/framework_bench/sgl_bench.py 1,4,8,16,32,64 100
"""
import sys
import time

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The quick brown fox"


def main():
    import sglang as sgl

    batches = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,4,8,16,32,64").split(",")]
    out = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    engine = sgl.Engine(model_path=MODEL, dtype="bfloat16", disable_radix_cache=True)
    sp = {"temperature": 0.0, "max_new_tokens": out, "ignore_eos": True}

    # warmup (graph capture + weight load), not timed
    engine.generate([PROMPT] * 2, sp)

    for B in batches:
        prompts = [PROMPT] * B
        t0 = time.perf_counter()
        outs = engine.generate(prompts, sp)
        dt = time.perf_counter() - t0
        # verify each sequence actually produced `out` tokens (else tok/s is misreported)
        ntok = sum(o["meta_info"]["completion_tokens"] for o in outs)
        tot = ntok / dt
        print(f"SGL B={B} TOTAL_TOK_S={tot:.1f} TOKENS={ntok} (expected {B*out}) "
              f"WALL_S={dt:.3f}", flush=True)

    engine.shutdown()


if __name__ == "__main__":
    main()
