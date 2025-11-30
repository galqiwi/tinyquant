# TinyQuant

A unified interface for neural network quantization. TinyQuant bridges the gap between quantization research and production inference by providing a simple, extensible framework that works across inference engines.

## The Problem

New quantization methods get published with research code, sometimes a HuggingFace integration. Production inference frameworks like vLLM need to implement each method separately. Research labs don't have resources to port to every framework. Result: most quantization methods never see production use.

## The Solution

Implement your quantization method once for TinyQuant. Use any quantization method from any inference framework that supports TinyQuant.

## Documentation

[How to implement new quantization method](./NEW_QUANT.md)

## Quick Start

```python
import torch
import transformers
from tinyquant.utils import quantize_matching_linear_layers

# Load model & tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B",
    device_map="cuda",
    dtype=torch.bfloat16,
)
tokenizer = transformers.AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

# One-line quantization: target attention q_proj layers via glob
quantize_matching_linear_layers(model, "nf4", "model.layers.*.self_attn.q_proj")

# Generate as usual
prompt = "Quantization for neural networks helps with "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galqiwi/tinyquant/blob/main/extra/huggingface-basic/llama_1b.ipynb)

## Installation

```bash
uv pip install tinyquant
```

For specific quantization methods, install their dependencies:

```bash
# For NF4 quantization
uv pip install bitsandbytes
```

## Features

**Unified serialization format**: Save and load quantized models regardless of quantization method. No more per-method serialization logic.

## Why TinyQuant?

**For researchers**: Implement once, run anywhere. Focus on the algorithm, not integration.

**For inference frameworks**: Integrate TinyQuant once, get access to all quantization methods.

**For users**: Try different quantization methods with the same API. Easy benchmarking and comparison.

## License

Apache 2.0


