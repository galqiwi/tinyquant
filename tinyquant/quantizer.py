from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Type
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
    def quantize(weight: torch.Tensor, bias: torch.Tensor, *args, **kwargs) -> "QuantizedLinear":
        pass

    @staticmethod
    @abstractmethod
    def forward(linear, input_):
        pass


_QUANTIZER_BY_NAME: Dict[str, Type[Quantizer]] = {}


def register_quantizer(quantizer_cls: Type[Quantizer]):
    _QUANTIZER_BY_NAME[quantizer_cls.name()] = quantizer_cls


def registered_quantizer(quantizer_cls: Type[Quantizer]):
    register_quantizer(quantizer_cls)
    return quantizer_cls


def get_quantizer(name: str) -> Type[Quantizer]:
    return _QUANTIZER_BY_NAME[name]


def quantize(name: str, weight: torch.Tensor, bias: torch.Tensor, *args, **kwargs):
    quantizer = get_quantizer(name)
    return quantizer.quantize(weight, bias, *args, **kwargs)