from dataclasses import dataclass
from typing import Literal, Optional

import torch
from . import rotation_utils, gptq_utils, utils

@dataclass
class QuaRotDataFreeConfig:
    w_bits: int = 4
    w_asym: bool = False
    w_groupsize: int = -1   # в оригинале RTN требует -1
    int8_down_proj: bool = False

    rotate: bool = True
    rotate_mode: Literal["hadamard", "random"] = "hadamard"
    fp32_had: bool = False


def apply_quarot_datafree_llama(
    model: torch.nn.Module,
    cfg: QuaRotDataFreeConfig,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:

    # 1) Rotation
    if cfg.rotate:
        # model_utils.get_model_type(model) внутри rotation_utils всё так же использует LLaMA_PATH
        rotation_utils.fuse_layer_norms(model)
        # придумай лёгкий "args"-объект вместо argparse
        class _Args:
            rotate_mode = cfg.rotate_mode
            fp32_had = cfg.fp32_had
        rotation_utils.rotate_model(model, _Args())

    # 2) RTN weight quantization
    class _ArgsRTN:
        w_bits = cfg.w_bits
        w_groupsize = cfg.w_groupsize
        w_asym = cfg.w_asym
        w_rtn = True
        int8_down_proj = cfg.int8_down_proj
        w_clip: bool = True

    dev = utils.DEV if device is None else torch.device(device)
    gptq_utils.rtn_fwrd(model, dev, _ArgsRTN())

    return model
