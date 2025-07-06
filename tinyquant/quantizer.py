from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Type, Any, Optional
import torch

if TYPE_CHECKING:
    from .quantized_linear import QuantizedLinear


class Quantizer(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def quantize(
        weight: torch.Tensor, bias: Optional[torch.Tensor], *args: Any, **kwargs: Any
    ) -> "QuantizedLinear":
        pass

    @staticmethod
    @abstractmethod
    def forward(linear: "QuantizedLinear", input_: torch.Tensor) -> torch.Tensor:
        pass


_QUANTIZER_BY_NAME: Dict[str, Type[Quantizer]] = {}


def register_quantizer(quantizer_cls: Type[Quantizer]) -> None:
    _QUANTIZER_BY_NAME[quantizer_cls.name()] = quantizer_cls


def registered_quantizer(quantizer_cls: Type[Quantizer]) -> Type[Quantizer]:
    register_quantizer(quantizer_cls)
    return quantizer_cls


def get_quantizer(name: str) -> Type[Quantizer]:
    return _QUANTIZER_BY_NAME[name]


def quantize(
    name: str,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> "QuantizedLinear":
    quantizer = get_quantizer(name)
    return quantizer.quantize(weight, bias, *args, **kwargs)
