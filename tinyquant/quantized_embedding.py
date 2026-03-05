from functools import cached_property
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from .quantized_linear import dequantize_meta, quantize_meta
from .quantizer import get_quantizer


class QuantizedEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tq_tensors = nn.ParameterDict()

    @classmethod
    def empty(cls) -> "QuantizedEmbedding":
        return QuantizedEmbedding()

    @classmethod
    def from_weights(
        cls,
        weights_dict: nn.ParameterDict,
        quantization_method: str,
        num_embeddings: int,
        embedding_dim: int,
        meta: Dict[str, Any],
    ) -> "QuantizedEmbedding":
        output = cls()

        tq_tensors = weights_dict
        if not isinstance(tq_tensors, nn.ParameterDict):
            raise TypeError(
                f"weights_dict must be nn.ParameterDict, got {type(tq_tensors)}"
            )

        for reserved_key in ("quantization_method", "num_embeddings", "embedding_dim"):
            if reserved_key in meta:
                raise ValueError(f"meta must not contain reserved key '{reserved_key}'")
        meta["quantization_method"] = quantization_method
        meta["num_embeddings"] = num_embeddings
        meta["embedding_dim"] = embedding_dim

        if "meta" in tq_tensors:
            raise ValueError("weights_dict must not contain reserved key 'meta'")
        tq_tensors["meta"] = nn.Parameter(quantize_meta(meta), requires_grad=False)

        output.tq_tensors = tq_tensors
        return output

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return get_quantizer(self.quantization_method).forward(self, indices)

    @cached_property
    def meta(self) -> Dict[str, Any]:
        if len(self.tq_tensors) == 0:
            raise RuntimeError("QuantizedEmbedding is not initialized")

        return dequantize_meta(self.tq_tensors["meta"])

    @cached_property
    def quantization_method(self) -> str:
        return str(self.meta["quantization_method"])

    @cached_property
    def num_embeddings(self) -> int:
        return int(self.meta["num_embeddings"])

    @cached_property
    def embedding_dim(self) -> int:
        return int(self.meta["embedding_dim"])

    @cached_property
    def shape(self) -> Tuple[int, int]:
        return self.num_embeddings, self.embedding_dim

    @property
    def weights_dict(self) -> Dict[str, nn.Parameter]:
        return {key: value for key, value in self.tq_tensors.items() if key != "meta"}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        if len(self.tq_tensors) != 0:
            raise RuntimeError(
                "load_state_dict called on already-initialized QuantizedEmbedding"
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
