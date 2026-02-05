import json
from functools import cached_property
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .quantizer import get_quantizer


def quantize_meta(meta: Dict[str, Any]) -> torch.Tensor:
    if not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict, got {type(meta)}")
    meta_tensor = torch.tensor(
        list(json.dumps(meta).encode("utf-8")),
        dtype=torch.uint8,
    )
    return meta_tensor


def dequantize_meta(meta_tensor: torch.Tensor) -> Dict[str, Any]:
    meta_bytes = meta_tensor.tolist()
    meta = bytes(meta_bytes).decode("utf-8")
    meta = json.loads(meta)
    if not isinstance(meta, dict):
        raise TypeError(f"expected meta to be a dict, got {type(meta)}")
    if not all(isinstance(key, str) for key in meta.keys()):
        raise TypeError("all meta keys must be strings")
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
        if not isinstance(tq_tensors, nn.ParameterDict):
            raise TypeError(
                f"weights_dict must be nn.ParameterDict, got {type(tq_tensors)}"
            )

        for reserved_key in ("quantization_method", "in_features", "out_features"):
            if reserved_key in meta:
                raise ValueError(f"meta must not contain reserved key '{reserved_key}'")
        meta["quantization_method"] = quantization_method
        meta["in_features"] = in_features
        meta["out_features"] = out_features

        if "meta" in tq_tensors:
            raise ValueError("weights_dict must not contain reserved key 'meta'")
        tq_tensors["meta"] = nn.Parameter(quantize_meta(meta), requires_grad=False)

        if "bias" in tq_tensors:
            raise ValueError("weights_dict must not contain reserved key 'bias'")
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError(f"bias must be a Tensor, got {type(bias)}")
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
        return self.out_features, self.in_features

    @property
    def weights_dict(self) -> Dict[str, nn.Parameter]:
        return {key: value for key, value in self.tq_tensors.items() if key != "meta"}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        if len(self.tq_tensors) != 0:
            raise RuntimeError(
                "load_state_dict called on already-initialized QuantizedLinear"
            )

        prefix = "tq_tensors."
        for key, value_tensor in state_dict.items():
            if not key.startswith(prefix):
                raise ValueError(f"unexpected key '{key}', expected prefix '{prefix}'")
            param_name = key[len(prefix) :]
            self.tq_tensors[param_name] = nn.Parameter(
                torch.empty_like(value_tensor),
                requires_grad=False,
            )

        return super().load_state_dict(state_dict, strict=strict)
