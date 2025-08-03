import torch
import torch.nn as nn
from functools import cached_property
import json
from typing import Dict, Any, Mapping, Optional
from .quantizer import get_quantizer


def quantize_meta(meta: Dict[str, Any]) -> torch.Tensor:
    assert isinstance(meta, dict)
    meta_tensor = torch.tensor(
        list(json.dumps(meta).encode('utf-8')),
        dtype=torch.uint8,
    )
    return meta_tensor


def dequantize_meta(meta_tensor: torch.Tensor) -> Dict[str, Any]:
    meta_bytes = meta_tensor.tolist()
    meta = bytes(meta_bytes).decode('utf-8')
    meta = json.loads(meta)
    assert isinstance(meta, dict)
    assert all(isinstance(key, str) for key in meta.keys())
    return meta


class QuantizedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights_dict = nn.ParameterDict()

    @classmethod
    def empty(cls) -> "QuantizedLinear":
        return QuantizedLinear()

    @classmethod
    def from_weights(
        cls, weights_dict: nn.ParameterDict, bias: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> "QuantizedLinear":
        output = cls()
        assert 'quantization_method' in meta

        assert 'in_features' in meta
        assert isinstance(meta['in_features'], int)
        assert 'out_features' in meta
        assert isinstance(meta['out_features'], int)

        assert 'meta' not in weights_dict
        assert isinstance(weights_dict, nn.ParameterDict)
        weights_dict['meta'] = nn.Parameter(
            quantize_meta(meta), requires_grad=False
        )

        assert 'bias' not in weights_dict
        if bias is not None:
            assert isinstance(bias, torch.Tensor)
            weights_dict['bias'] = nn.Parameter(
                bias, requires_grad=False
            )

        output.weights_dict = weights_dict
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return get_quantizer(self.quantization_method).forward(self, x)

    @cached_property
    def meta(self) -> Dict[str, Any]:
        if len(self.weights_dict) == 0:
            raise RuntimeError("QuantizedLinear is not initialized")

        return dequantize_meta(self.weights_dict['meta'])

    @cached_property
    def quantization_method(self) -> str:
        return str(self.meta['quantization_method'])

    @cached_property
    def in_features(self) -> int:
        return int(self.meta['in_features'])

    @cached_property
    def out_features(self) -> int:
        return int(self.meta['out_features'])

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        assert len(self.weights_dict) == 0

        prefix = 'weights_dict.'
        for key, value_tensor in state_dict.items():
            assert key.startswith(prefix)
            param_name = key[len(prefix):]
            self.weights_dict[param_name] = nn.Parameter(
                torch.empty_like(value_tensor),
                requires_grad=False,
            )

        return super().load_state_dict(state_dict, strict=strict)
