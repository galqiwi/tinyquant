from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer
from tinyquant.quantized_linear import QuantizedLinear
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@registered_quantizer
class NF4Quantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "nf4"

    @staticmethod
    def quantize(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> "QuantizedLinear":
        import bitsandbytes.functional

        out_features, in_features = weight.shape

        quantized_weight, quant_state = bitsandbytes.functional.quantize_nf4(weight, blocksize=64)

        return QuantizedLinear.from_weights(
            nn.ParameterDict({
                'quantized_weight': nn.Parameter(quantized_weight, requires_grad=False),
                'absmax': nn.Parameter(quant_state.absmax, requires_grad=False),
                'in_features': in_features,
                'out_features': out_features,
            }),
            bias,
            {'quantization_method': NF4Quantizer.name(), 'shape': tuple(weight.shape)},
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        import bitsandbytes.functional

        quant_state = bitsandbytes.functional.QuantState(
            absmax=linear.weights_dict['absmax'],
            shape=linear.meta['shape'],
            dtype=input_.dtype,
            blocksize=64,
            quant_type="nf4"
        )

        dequantized_weights = bitsandbytes.functional.dequantize_nf4(
            linear.weights_dict['quantized_weight'],
            quant_state=quant_state,
        )

        return F.linear(
            input_, dequantized_weights, linear.weights_dict.get('bias', None)
        )
