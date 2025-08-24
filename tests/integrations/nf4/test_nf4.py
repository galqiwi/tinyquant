from tinyquant.quantizer import quantize
from tinyquant.quantized_linear import QuantizedLinear

import pytest

import torch
import torch.nn as nn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4(dtype):
    in_features, out_features = 32, 20

    original = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device="cuda",
        bias=False,
    )

    quantized = quantize("nf4", original.weight, original.bias)

    dequantized = quantized(torch.eye(in_features, dtype=dtype, device="cuda")).T

    assert (
        torch.linalg.norm(dequantized - original.weight)
        / torch.linalg.norm(original.weight)
        < 0.15
    )
