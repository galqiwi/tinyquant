import torch
import fnmatch
from tinyquant.quantizer import quantize
from typing import Any, Iterable
from tqdm import tqdm


def _split_module_path(module_path: str) -> tuple[str | None, str]:
    parts = module_path.rsplit(".", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def quantize_all_linear_layers(
    model: torch.nn.Module, method_name: str, verbose: bool = False, *args: Any, **kwargs: Any
) -> None:
    linear_paths = []
    for module_path, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_paths.append(module_path)

    linear_paths_iter: Iterable[str]
    if verbose:
        linear_paths_iter = tqdm(linear_paths)
    else:
        linear_paths_iter = linear_paths

    for linear_path in linear_paths_iter:
        parent_name, linear_name = _split_module_path(linear_path)

        parent = model if parent_name is None else model.get_submodule(parent_name)
        linear = getattr(parent, linear_name)

        setattr(
            parent,
            linear_name,
            quantize(method_name, linear.weight, linear.bias, *args, **kwargs),
        )

        del parent, linear


def quantize_matching_linear_layers(
    model: torch.nn.Module,
    method_name: str,
    pattern: str,
    verbose: bool = False,
    *args: Any,
    **kwargs: Any,
) -> None:
    linear_paths = []
    for module_path, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if fnmatch.fnmatch(module_path, pattern):
                linear_paths.append(module_path)

    linear_paths_iter: Iterable[str]
    if verbose:
        linear_paths_iter = tqdm(linear_paths)
    else:
        linear_paths_iter = linear_paths

    for linear_path in linear_paths_iter:
        parent_name, linear_name = _split_module_path(linear_path)

        parent = model if parent_name is None else model.get_submodule(parent_name)
        linear = getattr(parent, linear_name)

        setattr(
            parent,
            linear_name,
            quantize(method_name, linear.weight, linear.bias, *args, **kwargs),
        )

        del parent, linear
