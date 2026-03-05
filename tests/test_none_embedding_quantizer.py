import pytest
import torch
import torch.nn as nn

from tinyquant.quantized_embedding import QuantizedEmbedding
from tinyquant.quantizer import quantize


def test_full_precision_embedding_quantizer() -> None:
    original = nn.Embedding(num_embeddings=100, embedding_dim=64)
    quantized = quantize("none_embedding", original.weight, None)

    indices = torch.tensor([0, 5, 42, 99])

    torch.testing.assert_close(quantized(indices), original(indices))


def test_load_full_precision_embedding() -> None:
    original = nn.Embedding(num_embeddings=100, embedding_dim=64)
    quantized = QuantizedEmbedding()
    quantized.load_state_dict(
        quantize("none_embedding", original.weight, None).state_dict()
    )

    indices = torch.tensor([0, 5, 42, 99])

    torch.testing.assert_close(quantized(indices), original(indices))


def test_embedding_multidim_indices() -> None:
    original = nn.Embedding(num_embeddings=50, embedding_dim=32)
    quantized = quantize("none_embedding", original.weight, None)

    indices = torch.tensor([[0, 1, 2], [10, 20, 30]])

    torch.testing.assert_close(quantized(indices), original(indices))
