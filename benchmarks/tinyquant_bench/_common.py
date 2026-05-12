import argparse
import json
from typing import Any, Dict, Tuple

import torch


_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    return _DTYPES[dtype_str]


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=parse_dtype(args.dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return model, tokenizer


def apply_quantization(model: torch.nn.Module, args: argparse.Namespace) -> None:
    backend = args.backend
    method = args.method

    if backend == "none":
        if method != "none":
            raise ValueError(
                f"--backend none requires --method none, got --method {method}"
            )
        return

    method_kwargs: Dict[str, Any] = json.loads(args.method_kwargs)
    embedding_method_kwargs: Dict[str, Any] = json.loads(args.embedding_method_kwargs)

    if backend == "tinyquant":
        from tinyquant_bench.tq_quantize import apply_tinyquant

        apply_tinyquant(
            model,
            method=method,
            pattern=args.pattern,
            method_kwargs=method_kwargs,
            embedding_method=args.embedding_method,
            embedding_pattern=args.embedding_pattern,
            embedding_method_kwargs=embedding_method_kwargs,
        )
        return

    if backend == "native":
        from tinyquant_bench.native_quantize import apply_native

        apply_native(
            model,
            method=method,
            pattern=args.pattern,
            method_kwargs=method_kwargs,
        )
        if args.embedding_method is not None:
            raise NotImplementedError(
                "native embedding quantization is not supported"
            )
        return

    raise ValueError(f"unknown backend: {backend}")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {
        "command": args.command,
        "model": args.model,
        "dtype": args.dtype,
        "method": args.method,
        "backend": args.backend,
        "pattern": args.pattern,
        "method_kwargs": json.loads(args.method_kwargs),
        "embedding_method": args.embedding_method,
        "embedding_pattern": args.embedding_pattern,
        "embedding_method_kwargs": json.loads(args.embedding_method_kwargs),
    }
    if args.command == "eval":
        cfg.update(
            {
                "tasks": args.tasks,
                "num_fewshot": args.num_fewshot,
                "batch_size": args.batch_size,
                "limit": args.limit,
            }
        )
    elif args.command == "speed":
        cfg.update(
            {
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "n_iters": args.n_iters,
                "n_warmup": args.n_warmup,
            }
        )
    return cfg
