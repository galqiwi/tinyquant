from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyquant.quantized_linear import QuantizedLinear
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class NoneQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "none"

    @staticmethod
    def quantize(
        weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> "QuantizedLinear":
        out_features, in_features = weight.shape

        return QuantizedLinear.from_weights(
            weights_dict=nn.ParameterDict(
                {
                    "weight": nn.Parameter(weight, requires_grad=False),
                }
            ),
            bias=bias,
            quantization_method=NoneQuantizer.name(),
            in_features=in_features,
            out_features=out_features,
            meta=dict(),
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input_, linear.weights_dict["weight"], linear.weights_dict.get("bias", None)
        )
