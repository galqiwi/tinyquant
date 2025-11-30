import torch

def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Простой FWHT (fast Walsh–Hadamard transform) на PyTorch.
    Ожидает последний размер степени двойки: (..., N), N = 2^k.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, f"last dim {n} is not power of 2"

    # приведём к 2D: [batch, n]
    x = x.reshape(-1, n)

    h = 1
    while h < n:
        # разбиваем на блоки из 2h
        x = x.view(-1, 2 * h)
        a = x[:, :h]
        b = x[:, h:]
        x = torch.cat([a + b, a - b], dim=1)
        h *= 2

    x = x.view(orig_shape)
    if scale != 1.0:
        x = x * scale
    return x
