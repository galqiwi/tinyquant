import argparse
from typing import Any, Dict, Optional

import torch

from tinyquant_bench.presets import resolve_tasks


def run_lm_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    task_names, default_fewshot = resolve_tasks(args.tasks)
    num_fewshot: Optional[int]
    if args.num_fewshot is not None:
        num_fewshot = args.num_fewshot
    else:
        num_fewshot = default_fewshot

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    results = simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )
    return _extract_metrics(results)


def _extract_metrics(results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if results is None:
        return {}
    summary: Dict[str, Any] = {}
    for task_name, task_result in results.get("results", {}).items():
        metrics = {}
        for key, value in task_result.items():
            if key == "alias":
                continue
            metrics[key] = value
        summary[task_name] = metrics
    return summary
