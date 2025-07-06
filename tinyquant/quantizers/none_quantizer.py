from tinyquant.quantizer import Quantizer, registered_quantizer
from tinyquant.quantized_linear import QuantizedLinear
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@registered_quantizer
class NoneQuantizer(Quantizer):
    @staticmethod
    def name() -> str:
        return "none"

    @staticmethod
    def quantize(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "QuantizedLinear":
        return QuantizedLinear.from_weights(nn.ParameterDict({
            'weight': weight,
        }), bias, {'quantization_method': NoneQuantizer.name()})

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input_, linear.weights_dict['weight'], linear.weights_dict['bias']
        )
