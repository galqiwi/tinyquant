from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import torch

if TYPE_CHECKING:
    from .quantized_embedding import QuantizedEmbedding
    from .quantized_linear import QuantizedLinear

    QuantizedModule = Union[QuantizedLinear, QuantizedEmbedding]


class Quantizer(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def forward(module: QuantizedModule, input_: torch.Tensor) -> torch.Tensor:
        pass


class DataFreeQuantizer(Quantizer, ABC):
    @staticmethod
    def quantize(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> QuantizedModule:
        raise NotImplementedError


_QUANTIZER_BY_NAME: Dict[str, Type[Quantizer]] = {}


def register_quantizer(quantizer_cls: Type[Quantizer]) -> None:
    _QUANTIZER_BY_NAME[quantizer_cls.name()] = quantizer_cls


def registered_quantizer(quantizer_cls: Type[Quantizer]) -> Type[Quantizer]:
    register_quantizer(quantizer_cls)
    return quantizer_cls


def get_quantizer(name: str) -> Type[Quantizer]:
    return _QUANTIZER_BY_NAME[name]


def quantize(
    method_name: str,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> QuantizedModule:
    quantizer = get_quantizer(method_name)
    assert issubclass(quantizer, DataFreeQuantizer)
    return quantizer.quantize(weight, bias, *args, **kwargs)
