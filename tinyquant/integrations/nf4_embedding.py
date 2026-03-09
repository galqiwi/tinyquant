from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyquant.quantized_embedding import QuantizedEmbedding
from tinyquant.quantizer import DataFreeQuantizer, registered_quantizer


@registered_quantizer
class NF4EmbeddingQuantizer(DataFreeQuantizer):
    @staticmethod
    def name() -> str:
        return "nf4_embedding"

    @staticmethod
    def quantize(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        block_size: int = 64,
    ) -> "QuantizedEmbedding":
        import bitsandbytes.functional

        num_embeddings, embedding_dim = weight.shape

        quantized_weight, quant_state = bitsandbytes.functional.quantize_nf4(
            weight, blocksize=block_size
        )

        return QuantizedEmbedding.from_weights(
            weights_dict=nn.ParameterDict(
                {
                    "quantized_weight": nn.Parameter(
                        quantized_weight, requires_grad=False
                    ),
                    "absmax": nn.Parameter(quant_state.absmax, requires_grad=False),
                }
            ),
            quantization_method=NF4EmbeddingQuantizer.name(),
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            meta={
                "block_size": block_size,
                "dtype": str(weight.dtype).replace("torch.", ""),
            },
        )

    @staticmethod
    def forward(embedding: "QuantizedEmbedding", indices: torch.Tensor) -> torch.Tensor:
        import bitsandbytes.functional

        num_embeddings = embedding.num_embeddings
        embedding_dim = embedding.embedding_dim
        block_size = embedding.meta["block_size"]
        dtype = getattr(torch, embedding.meta["dtype"])

        quantized_weight = embedding.weights_dict["quantized_weight"]
        absmax = embedding.weights_dict["absmax"]

        if embedding_dim % block_size == 0:
            return NF4EmbeddingQuantizer._forward_partial_dequantize(
                indices,
                quantized_weight,
                absmax,
                num_embeddings,
                embedding_dim,
                block_size,
                dtype,
            )

        quant_state = bitsandbytes.functional.QuantState(
            absmax=absmax,
            shape=(num_embeddings, embedding_dim),
            dtype=dtype,
            blocksize=block_size,
            quant_type="nf4",
        )
        dequantized = bitsandbytes.functional.dequantize_nf4(
            quantized_weight, quant_state=quant_state
        )
        return F.embedding(indices, dequantized)

    @staticmethod
    def _forward_partial_dequantize(
        indices: torch.Tensor,
        quantized_weight: torch.Tensor,
        absmax: torch.Tensor,
        num_embeddings: int,
        embedding_dim: int,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        import bitsandbytes.functional

        blocks_per_row = embedding_dim // block_size

        quantized_2d = quantized_weight.view(num_embeddings, embedding_dim // 2)
        absmax_2d = absmax.view(num_embeddings, blocks_per_row)

        input_shape = indices.shape
        flat_indices = indices.view(-1)

        selected_quantized = F.embedding(flat_indices, quantized_2d)
        selected_absmax = F.embedding(flat_indices, absmax_2d)

        n = flat_indices.numel()

        quant_state = bitsandbytes.functional.QuantState(
            absmax=selected_absmax.reshape(-1),
            shape=(n, embedding_dim),
            dtype=dtype,
            blocksize=block_size,
            quant_type="nf4",
        )
        dequantized = bitsandbytes.functional.dequantize_nf4(
            selected_quantized.reshape(-1), quant_state=quant_state
        )

        return dequantized.view(*input_shape, embedding_dim)
