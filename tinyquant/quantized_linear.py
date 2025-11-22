import json
from functools import cached_property
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .quantizer import get_quantizer


def quantize_meta(meta: Dict[str, Any]) -> torch.Tensor:
    assert isinstance(meta, dict)
    meta_tensor = torch.tensor(
        list(json.dumps(meta).encode("utf-8")),
        dtype=torch.uint8,
    )
    return meta_tensor


def dequantize_meta(meta_tensor: torch.Tensor) -> Dict[str, Any]:
    meta_bytes = meta_tensor.tolist()
    meta = bytes(meta_bytes).decode("utf-8")
    meta = json.loads(meta)
    assert isinstance(meta, dict)
    assert all(isinstance(key, str) for key in meta.keys())
    return meta


class QuantizedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tq_tensors = nn.ParameterDict()

    @classmethod
    def empty(cls) -> "QuantizedLinear":
        return QuantizedLinear()

    @classmethod
    def from_weights(
        cls,
        weights_dict: nn.ParameterDict,
        bias: Optional[torch.Tensor],
        quantization_method: str,
        in_features: int,
        out_features: int,
        meta: Dict[str, Any],
    ) -> "QuantizedLinear":
        output = cls()

        tq_tensors = weights_dict
        assert isinstance(tq_tensors, nn.ParameterDict)

        assert "quantization_method" not in meta
        meta["quantization_method"] = quantization_method

        assert "in_features" not in meta
        meta["in_features"] = in_features

        assert "out_features" not in meta
        meta["out_features"] = out_features

        assert "meta" not in tq_tensors
        tq_tensors["meta"] = nn.Parameter(quantize_meta(meta), requires_grad=False)

        assert "bias" not in tq_tensors
        if bias is not None:
            assert isinstance(bias, torch.Tensor)
            tq_tensors["bias"] = nn.Parameter(bias, requires_grad=False)

        output.tq_tensors = tq_tensors
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return get_quantizer(self.quantization_method).forward(self, x)

    @cached_property
    def meta(self) -> Dict[str, Any]:
        if len(self.tq_tensors) == 0:
            raise RuntimeError("QuantizedLinear is not initialized")

        return dequantize_meta(self.tq_tensors["meta"])

    @cached_property
    def quantization_method(self) -> str:
        return str(self.meta["quantization_method"])

    @cached_property
    def in_features(self) -> int:
        return int(self.meta["in_features"])

    @cached_property
    def out_features(self) -> int:
        return int(self.meta["out_features"])

    @cached_property
    def shape(self) -> Tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def weights_dict(self) -> Dict[str, nn.Parameter]:
        return {key: value for key, value in self.tq_tensors.items() if key != "meta"}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        assert len(self.tq_tensors) == 0

        prefix = "tq_tensors."
        for key, value_tensor in state_dict.items():
            assert key.startswith(prefix)
            param_name = key[len(prefix) :]
            self.tq_tensors[param_name] = nn.Parameter(
                torch.empty_like(value_tensor),
                requires_grad=False,
            )

        return super().load_state_dict(state_dict, strict=strict)
