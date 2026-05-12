import argparse
import statistics
import time
from typing import List

import torch

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


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _measure_forward_ms(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    n_warmup: int,
    n_iters: int,
) -> List[float]:
    use_cuda = torch.cuda.is_available() and input_ids.is_cuda
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids)
            if use_cuda:
                torch.cuda.synchronize()

        timings_ms: List[float] = []
        if use_cuda:
            for _ in range(n_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(input_ids)
                end.record()
                torch.cuda.synchronize()
                timings_ms.append(start.elapsed_time(end))
        else:
            for _ in range(n_iters):
                t0 = time.perf_counter()
                model(input_ids)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
    return timings_ms


def run_speed(args: argparse.Namespace) -> int:
    t0 = time.time()
    reset_gpu_peak()

    model, _tokenizer = load_model_and_tokenizer(args)
    bytes_before = measure_params_bytes(model)
    apply_quantization(model, args)
    bytes_after = measure_params_bytes(model)

    device = next(model.parameters()).device
    vocab_size = int(getattr(model.config, "vocab_size"))
    input_ids = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )

    timings_ms = _measure_forward_ms(
        model,
        input_ids,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
    )

    speed_stats = {
        "median_ms": statistics.median(timings_ms),
        "p10_ms": _percentile(timings_ms, 0.1),
        "p90_ms": _percentile(timings_ms, 0.9),
        "mean_ms": statistics.fmean(timings_ms),
        "stdev_ms": statistics.stdev(timings_ms) if len(timings_ms) >= 2 else 0.0,
        "n_iters": args.n_iters,
        "n_warmup": args.n_warmup,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "device": str(device),
    }

    out = {
        "config": build_config(args),
        "size": size_summary(bytes_before, bytes_after),
        "gpu_peak_bytes": gpu_peak_bytes(),
        "speed": speed_stats,
        "wall_time_sec": time.time() - t0,
    }
    write_output(args.output, out)
    maybe_log_wandb(
        args.wandb_project,
        args.wandb_name,
        config=out["config"],
        metrics={"size": out["size"], "speed": speed_stats},
    )
    return 0
