# tinyquant benchmarks

Measures quality, size, and inference speed of `tinyquant`-quantized models against fp16/bf16 baselines and native (non-`tinyquant`) implementations of the same methods.

## Install

The package is a `uv` workspace member. From the repo root:

```bash
# Core + tinyquant (no quantization backends)
uv sync --package tinyquant-bench

# With specific backends
uv sync --package tinyquant-bench --extra nf4
uv sync --package tinyquant-bench --extra hqq
uv sync --package tinyquant-bench --extra wandb

# Everything
uv sync --package tinyquant-bench --extra all
```

`bitsandbytes` (nf4), `hqq`, `higgs-kernels`, and `wandb` are CUDA-dependent or optional, hence the extras.

## Usage

Two subcommands: `eval` (quality via `lm-eval-harness`) and `speed` (forward-pass latency).

### Quality

```bash
# Baseline fp16/bf16, quick smoke
uv run python -m tinyquant_bench eval \
    --model unsloth/Llama-3.2-1B \
    --dtype bfloat16 \
    --method none \
    --backend none \
    --tasks quick \
    --output baseline.json

# nf4 via tinyquant, full zero-shot suite
uv run python -m tinyquant_bench eval \
    --model unsloth/Llama-3.2-1B \
    --method nf4 \
    --backend tinyquant \
    --pattern 'model.layers.*.self_attn.q_proj' \
    --method-kwargs '{"block_size": 64}' \
    --tasks zero_shot \
    --batch-size 8 \
    --output tq_nf4_zero_shot.json

# Same method via native bitsandbytes (no tinyquant wrapper) for direct comparison
uv run python -m tinyquant_bench eval \
    --model unsloth/Llama-3.2-1B \
    --method nf4 \
    --backend native \
    --pattern 'model.layers.*.self_attn.q_proj' \
    --method-kwargs '{"block_size": 64}' \
    --tasks zero_shot \
    --batch-size 8 \
    --output native_nf4_zero_shot.json
```

Task presets:

| preset      | tasks                                                | shots | what it measures        |
| ----------- | ---------------------------------------------------- | ----- | ----------------------- |
| `quick`     | `arc_easy`                                           | 1     | smoke (~1 min)          |
| `zero_shot` | `winogrande, piqa, hellaswag, arc_easy, arc_challenge` | 1     | standard zero-shot      |
| `mmlu`      | `mmlu`                                               | 5     | reasoning across 57 subjects |
| `ppl`       | `wikitext`                                           | 0     | perplexity on Wikitext-2 |

Or pass a csv: `--tasks arc_easy,piqa,hellaswag`.

### Speed

```bash
uv run python -m tinyquant_bench speed \
    --model unsloth/Llama-3.2-1B \
    --method nf4 \
    --backend tinyquant \
    --pattern 'model.layers.*.self_attn.q_proj' \
    --batch-size 1 \
    --seq-len 128 \
    --n-iters 100 \
    --n-warmup 10 \
    --output speed_tq_nf4.json
```

The intended use of `speed` is comparing `--backend tinyquant` against `--backend native` on the same `--method` and `--pattern` — that isolates the overhead the `QuantizedLinear` wrapper adds over the underlying kernel.

## Backends

| `--backend`  | what it does                                                  |
| ------------ | ------------------------------------------------------------- |
| `tinyquant`  | uses `tinyquant.utils.quantize_matching_linear_layers` → `QuantizedLinear` |
| `native`     | replaces `nn.Linear` with `bnb.nn.Linear4bit` / `HQQLinear` directly |
| `none`       | no quantization, fp16/bf16 baseline                           |

`--backend native --method higgs` is not supported in v1.

## Output

JSON with three sections — `config`, `size`, `results` (eval) or `speed` (speed).

```json
{
  "config": {"model": "...", "method": "nf4", "backend": "tinyquant", ...},
  "size": {
    "params_bytes_baseline": 2500000000,
    "params_bytes": 1900000000,
    "compression_ratio": 1.316
  },
  "gpu_peak_bytes": 3100000000,
  "results": {
    "arc_easy": {"acc,none": 0.71, "acc_stderr,none": 0.012}
  },
  "wall_time_sec": 123.4
}
```

Optional `--wandb-project`/`--wandb-name` logs the same dict to Weights & Biases.

## Smoke

`run_all.py` runs a tiny matrix end-to-end (requires CUDA + the `nf4` extra):

```bash
./benchmarks/run_all.py
```
