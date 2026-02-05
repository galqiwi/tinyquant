from typing import Optional

import torch
import torch.nn as nn

from tinyquant.quantized_linear import QuantizedLinear
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class HIGGSQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "higgs"

    @staticmethod
    def quantize(
        weight: torch.Tensor, bias: Optional[torch.Tensor], group_size: int = 1024
    ) -> "QuantizedLinear":
        import higgs_kernels
        import higgs_kernels.grids
        import higgs_kernels.linear

        if not weight.is_cuda:
            raise ValueError("HIGGS quantization requires CUDA tensors")
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"HIGGS requires float16 or bfloat16, got {weight.dtype}")

        if bias is not None and weight.device != bias.device:
            raise ValueError(
                f"weight device {weight.device} != bias device {bias.device}"
            )
        if bias is not None and weight.dtype != bias.dtype:
            raise ValueError(f"weight dtype {weight.dtype} != bias dtype {bias.dtype}")

        out_features, in_features = weight.shape

        grid = higgs_kernels.grids.load_optimal_grid_2_256(
            device="cpu", dtype=weight.dtype
        ).to(weight.device)
        quantized, scales = higgs_kernels.linear.higgs_quantize_linear_2_256(
            weight, grid, group_size
        )

        return QuantizedLinear.from_weights(
            weights_dict=nn.ParameterDict(
                {
                    "grid": nn.Parameter(grid, requires_grad=False),
                    "quantized": nn.Parameter(quantized, requires_grad=False),
                    "scales": nn.Parameter(scales, requires_grad=False),
                }
            ),
            bias=bias,
            quantization_method=HIGGSQuantizer.name(),
            in_features=in_features,
            out_features=out_features,
            meta={"group_size": group_size},
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        import higgs_kernels.linear

        return higgs_kernels.linear.higgs_matmul_linear_2_256(
            input_,
            linear.weights_dict["quantized"],
            linear.weights_dict["scales"],
            linear.weights_dict["grid"],
            linear.meta["group_size"],
        )
