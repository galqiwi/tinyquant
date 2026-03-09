from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyquant.quantized_embedding import QuantizedEmbedding
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class NoneEmbeddingQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "none_embedding"

    @staticmethod
    def quantize(
        weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> "QuantizedEmbedding":
        num_embeddings, embedding_dim = weight.shape

        return QuantizedEmbedding.from_weights(
            weights_dict=nn.ParameterDict(
                {
                    "weight": nn.Parameter(weight, requires_grad=False),
                }
            ),
            quantization_method=NoneEmbeddingQuantizer.name(),
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            meta=dict(),
        )

    @staticmethod
    def forward(embedding: "QuantizedEmbedding", indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, embedding.weights_dict["weight"])
