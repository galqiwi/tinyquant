# TinyQuant

A unified interface for neural network quantization. TinyQuant bridges the gap between quantization research and production inference by providing a simple, extensible framework that works across inference engines.

## The Problem

New quantization methods get published with research code, sometimes a HuggingFace integration. Production inference frameworks like vLLM need to implement each method separately. Research labs don't have resources to port to every framework. Result: most quantization methods never see production use.

## The Solution

Implement your quantization method once for TinyQuant. Use any quantization method from any inference framework that supports TinyQuant.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM
from tinyquant.utils import quantize_matching_linear_layers

# Load any PyTorch model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Quantize with one line - use pattern matching to target specific layers
quantize_matching_linear_layers(model, "nf4", "model.layers.*.self_attn.q_proj")

# Or quantize all linear layers
from tinyquant.utils import quantize_all_linear_layers
quantize_all_linear_layers(model, "higgs", verbose=True)

# Model works exactly as before, but uses less memory
output = model.generate(...)
```

## Installation

```bash
pip install tinyquant
```

For specific quantization methods, install their dependencies:

```bash
# For NF4 quantization
pip install bitsandbytes
```

## Features

**Unified serialization format**: Save and load quantized models regardless of quantization method. No more per-method serialization logic.

## Why TinyQuant?

**For researchers**: Implement once, run anywhere. Focus on the algorithm, not integration.

**For inference frameworks**: Integrate TinyQuant once, get access to all quantization methods.

**For users**: Try different quantization methods with the same API. Easy benchmarking and comparison.

## License

Apache 2.0


