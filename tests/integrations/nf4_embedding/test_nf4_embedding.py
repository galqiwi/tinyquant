import pytest
import torch
import torch.nn as nn

from tinyquant.quantizer import quantize


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4_embedding(dtype):
    num_embeddings, embedding_dim = 128, 64
    block_size = 64

    original = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        dtype=dtype,
        device="cuda",
    )

    quantized = quantize("nf4_embedding", original.weight, None, block_size=block_size)

    indices = torch.arange(num_embeddings, device="cuda")
    original_output = original(indices)
    quantized_output = quantized(indices)

    assert (
        torch.linalg.norm(quantized_output - original_output)
        / torch.linalg.norm(original_output)
        < 0.15
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4_embedding_multidim_indices(dtype):
    num_embeddings, embedding_dim = 128, 128

    original = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        dtype=dtype,
        device="cuda",
    )

    quantized = quantize("nf4_embedding", original.weight, None)

    indices = torch.tensor([[0, 1, 2], [10, 20, 30]], device="cuda")
    original_output = original(indices)
    quantized_output = quantized(indices)

    assert (
        torch.linalg.norm(quantized_output - original_output)
        / torch.linalg.norm(original_output)
        < 0.15
    )
