# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/13 8:49
from typing import Dict, Type
import torch
from ..lycoris.lycoris import PetlLycorisBase
from ....layers.petl.loha.layer import Conv2d, Linear, LoHaLayer


class LoHaModule(PetlLycorisBase):
    """
    Creates Low-Rank Hadamard Product model from a pretrained model. The method is partially described in
    https://arxiv.org/abs/2108.06098 Current implementation heavily borrows from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`LoHaConfig`]): The configuration of the LoHa model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The LoHa model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoHaModel, LoHaConfig

        >>> config_te = LoHaConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = LoHaConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ...     use_effective_conv2d=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoHaModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`LoHaConfig`]): The configuration of the LoHa model.
    """

    prefix: str = "hada_"
    layers_mapping: Dict[Type[torch.nn.Module], Type[LoHaLayer]] = {
        torch.nn.Conv2d: Conv2d,
        torch.nn.Linear: Linear,
    }
