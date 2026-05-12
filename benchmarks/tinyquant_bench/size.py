from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def measure_params_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def measure_buffers_bytes(model: nn.Module) -> int:
    return sum(b.numel() * b.element_size() for b in model.buffers())


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def size_summary(
    bytes_before: int,
    bytes_after: int,
    buffers_after: Optional[int] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "params_bytes_baseline": bytes_before,
        "params_bytes": bytes_after,
        "compression_ratio": (bytes_before / bytes_after) if bytes_after > 0 else None,
    }
    if buffers_after is not None:
        summary["buffers_bytes"] = buffers_after
    return summary
