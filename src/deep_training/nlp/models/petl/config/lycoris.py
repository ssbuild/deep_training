# -*- coding: utf-8 -*-
# @Time:  10:49
# @Author: tk
# @File：lycoris_config
from dataclasses import dataclass, field
from typing import Union, Optional, List
from .petl import PetlConfig

@dataclass
class LycorisConfig(PetlConfig):
    r"""
    A base config for LyCORIS like adapters
    """
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )