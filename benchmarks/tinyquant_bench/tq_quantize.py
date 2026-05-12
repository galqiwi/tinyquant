from typing import Any, Dict, Optional

import torch.nn as nn

from tinyquant.utils import (
    quantize_matching_embedding_layers,
    quantize_matching_linear_layers,
)


def apply_tinyquant(
    model: nn.Module,
    *,
    method: str,
    pattern: str,
    method_kwargs: Dict[str, Any],
    embedding_method: Optional[str] = None,
    embedding_pattern: Optional[str] = None,
    embedding_method_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    quantize_matching_linear_layers(model, method, pattern, **method_kwargs)

    if embedding_method is not None:
        if embedding_pattern is None:
            raise ValueError(
                "--embedding-pattern is required when --embedding-method is set"
            )
        quantize_matching_embedding_layers(
            model,
            embedding_method,
            embedding_pattern,
            **(embedding_method_kwargs or {}),
        )
