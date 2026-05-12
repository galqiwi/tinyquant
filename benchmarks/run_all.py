#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Smoke runner for tinyquant benchmarks.

Runs a tiny matrix on a small Llama model to verify the pipeline works
end-to-end. Requires CUDA + the [nf4] extra (bitsandbytes).
"""

import os
import subprocess
import sys

PKG_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL = os.environ.get("TQ_BENCH_SMOKE_MODEL", "unsloth/Llama-3.2-1B")

SMOKE_CONFIGS = [
    [
        "eval",
        "--model", MODEL,
        "--method", "none",
        "--backend", "none",
        "--tasks", "quick",
        "--limit", "5",
        "--batch-size", "1",
        "--output", "smoke_baseline.json",
    ],
    [
        "eval",
        "--model", MODEL,
        "--method", "nf4",
        "--backend", "tinyquant",
        "--pattern", "model.layers.*.self_attn.q_proj",
        "--method-kwargs", '{"block_size": 64}',
        "--tasks", "quick",
        "--limit", "5",
        "--batch-size", "1",
        "--output", "smoke_tq_nf4.json",
    ],
    [
        "speed",
        "--model", MODEL,
        "--method", "none",
        "--backend", "none",
        "--batch-size", "1",
        "--seq-len", "64",
        "--n-iters", "10",
        "--n-warmup", "3",
        "--output", "smoke_speed_baseline.json",
    ],
]


def main() -> int:
    for i, config in enumerate(SMOKE_CONFIGS, 1):
        print(f"\n=== smoke {i}/{len(SMOKE_CONFIGS)}: {' '.join(config[:4])} ===")
        cmd = ["uv", "run", "python", "-m", "tinyquant_bench", *config]
        result = subprocess.run(cmd, cwd=PKG_PATH)
        if result.returncode != 0:
            print(f"smoke {i} failed (exit {result.returncode})")
            return result.returncode
    print("\nall smoke configs passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
