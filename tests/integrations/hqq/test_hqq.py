import pytest
import torch
import torch.nn as nn

from tinyquant.quantizer import quantize


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nhqq(dtype):
    in_features, out_features = 32, 20

    original = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device="cuda",
        bias=False,
    )

    quantized = quantize("hqq", original.weight, original.bias)

    dequantized = quantized(torch.eye(in_features, dtype=dtype, device="cuda")).T

    assert (
        torch.linalg.norm(dequantized - original.weight)
        / torch.linalg.norm(original.weight)
        < 0.15
    )
