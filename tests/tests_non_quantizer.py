from tinyquant.quantizer import quantize
from tinyquant.quantized_linear import QuantizedLinear

import torch
import torch.nn as nn


def test_full_precision_quantizer() -> None:
    original = nn.Linear(in_features=10, out_features=20)
    quantized = quantize('full_precision', original.weight, original.bias)

    input_ = torch.randn(10)

    torch.testing.assert_close(quantized(input_), original(input_))


def test_load_full_precision_linear() -> None:
    original = nn.Linear(in_features=10, out_features=20)
    quantized = QuantizedLinear()
    quantized.load_state_dict(
        quantize('full_precision', original.weight, original.bias).state_dict()
    )

    input_ = torch.randn(10)

    torch.testing.assert_close(quantized(input_), original(input_))
