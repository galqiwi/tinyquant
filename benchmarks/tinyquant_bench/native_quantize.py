import fnmatch
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn


def _iter_linear_modules(
    model: nn.Module, pattern: str
) -> Iterator[Tuple[str, nn.Linear]]:
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear) and fnmatch.fnmatch(path, pattern):
            yield path, module


def _replace_submodule(model: nn.Module, path: str, new_module: nn.Module) -> None:
    if "." in path:
        parent_path, name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
    else:
        parent, name = model, path
    setattr(parent, name, new_module)


def _quantize_nf4_native(
    model: nn.Module,
    pattern: str,
    *,
    block_size: int = 64,
) -> None:
    import bitsandbytes as bnb

    for path, old in list(_iter_linear_modules(model, pattern)):
        device = old.weight.device
        dtype = old.weight.dtype
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            compute_dtype=dtype,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        new.weight = bnb.nn.Params4bit(
            data=old.weight.data.contiguous(),
            requires_grad=False,
            quant_type="nf4",
            blocksize=block_size,
        )
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.clone(), requires_grad=False)
        new = new.to(device)
        _replace_submodule(model, path, new)
        del old


def _quantize_hqq_native(
    model: nn.Module,
    pattern: str,
    *,
    nbits: int = 4,
    group_size: int = 64,
    **extra_cfg: Any,
) -> None:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, **extra_cfg)
    for path, old in list(_iter_linear_modules(model, pattern)):
        new = HQQLinear(
            linear_layer=old,
            quant_config=quant_config,
            compute_dtype=old.weight.dtype,
            device=old.weight.device,
            del_orig=False,
        )
        _replace_submodule(model, path, new)
        del old


def apply_native(
    model: nn.Module,
    *,
    method: str,
    pattern: str,
    method_kwargs: Dict[str, Any],
) -> None:
    if method == "nf4":
        _quantize_nf4_native(model, pattern, **method_kwargs)
        return
    if method == "hqq":
        _quantize_hqq_native(model, pattern, **method_kwargs)
        return
    if method == "higgs":
        raise NotImplementedError(
            "native HIGGS backend is not supported in v1; use --backend tinyquant"
        )
    if method == "none":
        raise ValueError(
            "--backend native --method none is meaningless; "
            "use --backend none --method none for baseline"
        )
    raise ValueError(f"unknown method for native backend: {method}")
