from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer
from tinyquant.quantized_linear import QuantizedLinear
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@registered_quantizer
class NoneQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "none"

    @staticmethod
    def quantize(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "QuantizedLinear":

        out_features, in_features = weight.shape

        return QuantizedLinear.from_weights(
            nn.ParameterDict({
                'weight': nn.Parameter(weight, requires_grad=False),
            }),
            bias, {
                'quantization_method': NoneQuantizer.name(),
                'in_features': in_features,
                'out_features': out_features,
            },
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input_, linear.weights_dict['weight'], linear.weights_dict.get('bias', None)
        )
