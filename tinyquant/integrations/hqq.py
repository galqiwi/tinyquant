from typing import Any, Optional

import torch
import torch.nn as nn

from tinyquant.quantized_linear import QuantizedLinear
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class HQQQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "hqq"

    @staticmethod
    def quantize(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        nbits: int = 4,
        group_size: int = 64,
        compute_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **extra_cfg: Any,
    ) -> "QuantizedLinear":
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

        out_features, in_features = weight.shape

        if device is None:
            device = weight.device

        if compute_dtype is None:
            compute_dtype = weight.dtype

        linear_base = nn.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            device=device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            linear_base.weight.copy_(weight)
            if bias is not None:
                linear_base.bias.copy_(bias)

        quant_config = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, **extra_cfg
        )

        hqq_layer = HQQLinear(
            linear_layer=linear_base,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            del_orig=True,
        )

        state_dict = hqq_layer.state_dict()
        state_dict_as_parameter_dict = nn.ParameterDict()
        for k, v in state_dict.items():
            state_dict_as_parameter_dict[k] = nn.Parameter(v, requires_grad=False)

        return QuantizedLinear.from_weights(
            weights_dict=state_dict_as_parameter_dict,
            bias=None,
            quantization_method=HQQQuantizer.name(),
            in_features=in_features,
            out_features=out_features,
            meta={
                "config": quant_config,
                "device": str(device),
                "compute_dtype": str(compute_dtype).replace("torch.", ""),
            },
        )

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        from hqq.core.quantize import HQQLinear

        hqq_state = {k: v.data for k, v in linear.weights_dict.items() if k != "meta"}

        hqq_cfg = linear.meta.get("meta", None)
        hqq_device = linear.meta.get("device", None)
        hqq_compute_dtype = linear.meta.get("compute_dtype", None)

        hqq_layer = HQQLinear(
            linear_layer=None,
            quant_config=hqq_cfg,
            compute_dtype=getattr(torch, hqq_compute_dtype),
            device=hqq_device,
            del_orig=True,
            initialize=False,
        )

        hqq_layer.load_state_dict(hqq_state)
        return hqq_layer(input_)
