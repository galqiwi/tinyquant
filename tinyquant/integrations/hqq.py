from typing import Optional, Dict, Any

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
        try:
            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        except Exception as e:
            raise ImportError(
                "HQQ is not installed. Please `pip install hqq>=0.2.1`."
            ) from e

        out_features, in_features = weight.shape
        device = device or weight.device
        compute_dtype = compute_dtype or weight.dtype

        base = nn.Linear(in_features, out_features, bias=bias is not None, device=device, dtype=weight.dtype)
        with torch.no_grad():
            base.weight.copy_(weight)
            if bias is not None:
                base.bias.copy_(bias)

        quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, **extra_cfg)

        hqq_layer = HQQLinear(
            linear_layer=base,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device,
            del_orig=True,
        )

        state = hqq_layer.state_dict()
        pd = nn.ParameterDict()
        for k, v in state.items():
            pd[k] = nn.Parameter(v, requires_grad=False)

        qlin = QuantizedLinear.from_weights(
            weights_dict=pd,
            bias=None,
            quantization_method=HQQQuantizer.name(),
            in_features=in_features,
            out_features=out_features,
            meta={
                    'config': quant_config,
                    'device': str(device),
                    'compute_dtype': str(compute_dtype).replace("torch.", ""),
                },
        )

        return qlin

    @staticmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        try:
            from hqq.core.quantize import HQQLinear
        except Exception as e:
            raise ImportError(
                "HQQ is not installed. Please `pip install hqq>=0.2.1`."
            ) from e
        hqq_state = {k: v.data for k, v in linear.weights_dict.items() if k != 'meta'}
        
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
