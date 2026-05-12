import argparse
import time
from typing import Any, Dict

from tinyquant_bench._common import (
    apply_quantization,
    build_config,
    load_model_and_tokenizer,
)
from tinyquant_bench.output import maybe_log_wandb, write_output
from tinyquant_bench.size import (
    gpu_peak_bytes,
    measure_params_bytes,
    reset_gpu_peak,
    size_summary,
)


def run_eval(args: argparse.Namespace) -> int:
    t0 = time.time()
    reset_gpu_peak()

    model, tokenizer = load_model_and_tokenizer(args)
    bytes_before = measure_params_bytes(model)

    apply_quantization(model, args)
    bytes_after = measure_params_bytes(model)

    eval_results: Dict[str, Any] = {}
    if args.tasks is not None:
        from tinyquant_bench.harness import run_lm_eval

        eval_results = run_lm_eval(model, tokenizer, args)

    out = {
        "config": build_config(args),
        "size": size_summary(bytes_before, bytes_after),
        "gpu_peak_bytes": gpu_peak_bytes(),
        "results": eval_results,
        "wall_time_sec": time.time() - t0,
    }
    write_output(args.output, out)
    maybe_log_wandb(
        args.wandb_project,
        args.wandb_name,
        config=out["config"],
        metrics={"size": out["size"], "results": eval_results},
    )
    return 0
