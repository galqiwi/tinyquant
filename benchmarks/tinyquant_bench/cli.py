import argparse
import sys
from typing import List, Optional


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for the model",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Quantization method name: none | nf4 | hqq | higgs | ...",
    )
    parser.add_argument(
        "--backend",
        default="tinyquant",
        choices=["tinyquant", "native", "none"],
        help="tinyquant: via QuantizedLinear; native: direct bnb/hqq replacement; "
        "none: no quantization (baseline)",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="fnmatch pattern over module paths to quantize",
    )
    parser.add_argument(
        "--method-kwargs",
        default="{}",
        help='JSON dict with kwargs passed to the quantizer, e.g. \'{"block_size": 64}\'',
    )
    parser.add_argument(
        "--embedding-method",
        default=None,
        help="Optional embedding quantizer name (e.g. nf4_embedding)",
    )
    parser.add_argument(
        "--embedding-pattern",
        default=None,
        help="fnmatch pattern for embeddings",
    )
    parser.add_argument(
        "--embedding-method-kwargs",
        default="{}",
        help="JSON dict with embedding quantizer kwargs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results to (stdout if omitted)",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="If set, log run to this wandb project",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="wandb run name",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyquant_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Quality benchmark via lm-eval-harness")
    _add_common_args(eval_p)
    eval_p.add_argument(
        "--tasks",
        default=None,
        help="Preset (quick|zero_shot|mmlu|ppl) or csv of lm-eval task names. "
        "If omitted, only size info is reported.",
    )
    eval_p.add_argument("--num-fewshot", type=int, default=None)
    eval_p.add_argument("--batch-size", type=int, default=1)
    eval_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (debug)",
    )

    speed_p = sub.add_parser("speed", help="Forward-pass latency microbenchmark")
    _add_common_args(speed_p)
    speed_p.add_argument("--batch-size", type=int, default=1)
    speed_p.add_argument("--seq-len", type=int, default=128)
    speed_p.add_argument("--n-iters", type=int, default=100)
    speed_p.add_argument("--n-warmup", type=int, default=10)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        from tinyquant_bench.eval import run_eval

        return run_eval(args)
    if args.command == "speed":
        from tinyquant_bench.speed import run_speed

        return run_speed(args)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
