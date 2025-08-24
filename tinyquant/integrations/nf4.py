from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyquant.quantized_linear import QuantizedLinear
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class NF4Quantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "nf4"

    @staticmethod
    def quantize(
        weight: torch.Tensor, bias: Optional[torch.Tensor], block_size: int = 64
    ) -> "QuantizedLinear":
        import bitsandbytes.functional

        out_features, in_features = weight.shape

        quantized_weight, quant_state = bitsandbytes.functional.quantize_nf4(
            weight, blocksize=block_size
        )

        return QuantizedLinear.from_weights(
            weights_dict=nn.ParameterDict(
                {
                    "quantized_weight": nn.Parameter(
                        quantized_weight, requires_grad=False
                    ),
                    "absmax": nn.Parameter(quant_state.absmax, requires_grad=False),
                }
            ),
            bias=bias,
            quantization_method=NF4Quantizer.name(),
            in_features=in_features,
            out_features=out_features,
            meta={"block_size": block_size},
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        import bitsandbytes.functional

        quant_state = bitsandbytes.functional.QuantState(
            absmax=linear.weights_dict["absmax"],
            shape=linear.shape,
            dtype=input_.dtype,
            blocksize=linear.meta["block_size"],
            quant_type="nf4",
        )

        dequantized_weights = bitsandbytes.functional.dequantize_nf4(
            linear.weights_dict["quantized_weight"],
            quant_state=quant_state,
        )

        return F.linear(
            input_, dequantized_weights, linear.weights_dict.get("bias", None)
        )
