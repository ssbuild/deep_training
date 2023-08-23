# -*- coding: utf-8 -*-
# @Time:  22:39
# @Author: tk
# @File：ia3
import re
import warnings
from dataclasses import asdict
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F
from .petl_model_internel import PetlModelAbstract
from ....layers.petl.ia3.ia3 import IA3Layer
from ....layers.petl.utils import transpose, is_bnb_available, _get_submodules, ModulesToSaveWrapper, \
    _is_valid_match
from transformers.pytorch_utils import Conv1D

if is_bnb_available():
    import bitsandbytes as bnb



class IA3Module(PetlModelAbstract):
    """
    Creates a Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3) model from a pretrained
    transformers model. The method is described in detail in https://arxiv.org/abs/2205.05638

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the (IA)^3 model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The (IA)^3 model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["k", "v", "w0"],
        ...     feedforward_modules=["w0"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ia3Config`]): The configuration of the (IA)^3 model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_module(ia3_config, adapter_name, target, **kwargs):
        bias = hasattr(target, "bias") and target.bias is not None
        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        is_feedforward = kwargs.pop("is_feedforward", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name,
                target.in_features,
                target.out_features,
                is_feedforward,
                bias=bias,
                **eightbit_kwargs,
            )
        else:
            #  Create a new Linear module with (IA)^3 parameters for torch.nn.Linear
            # or Conv1D modules
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = Linear(
                adapter_name, in_features, out_features, is_feedforward=is_feedforward, bias=bias, **kwargs
            )
        return new_module

    @staticmethod
    def _check_target_module_exists(ia3_config, key):
        if isinstance(ia3_config.target_modules, str):
            target_module_found = re.fullmatch(ia3_config.target_modules, key)
        else:
            target_module_found = any(_is_valid_match(key, target_key) for target_key in ia3_config.target_modules)
        return target_module_found

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if "ia3_" not in n:
                p.requires_grad = False

    def _create_and_replace(
        self,
        ia3_config,
        adapter_name,
        target,
        target_name,
        parent,
        **optionnal_kwargs,
    ):
        loaded_in_8bit = optionnal_kwargs["loaded_in_8bit"]
        current_key = optionnal_kwargs["current_key"]

        # check if target module is in feedforward_modules
        if isinstance(ia3_config.feedforward_modules, str):
            is_feedforward = re.fullmatch(ia3_config.feedforward_modules, current_key)
        else:
            is_feedforward = any(current_key.endswith(target_key) for target_key in ia3_config.feedforward_modules)

        kwargs = {
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
            "init_ia3_weights": ia3_config.init_ia3_weights,
            "loaded_in_8bit": loaded_in_8bit,
            "is_feedforward": is_feedforward,
        }

        if isinstance(target, IA3Layer):
            target.update_layer(
                adapter_name,
                ia3_config.init_ia3_weights,
            )
        else:
            new_module = self._create_new_module(ia3_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if child.bias is not None:
            new_module.bias = child.bias
        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ia3_" in name:
                module.to(child.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                module.disable_adapters = False if enabled else True
            elif isinstance(module, ModulesToSaveWrapper):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, IA3Layer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def _prepare_adapter_config(self, peft_config, model_config):
        # if peft_config.target_modules is None:
        #     if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
        #         raise ValueError("Please specify `target_modules` in `peft_config`")
        #     peft_config.target_modules = TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]]
        # if peft_config.feedforward_modules is None:
        #     if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING:
        #         raise ValueError("Please specify `feedforward_modules` in `peft_config`")
        #     peft_config.feedforward_modules = TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[
        #         model_config["model_type"]
        #     ]
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the (IA)^3 layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging ia3 layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge ia3 layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "ia3" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, IA3Layer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


# Below code is based on https://github.com/microsoft/lora/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, IA3Layer):
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        **kwargs,
    ):
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_ia3_weights)
        self.active_adapter = adapter_name

        self.is_feedforward = is_feedforward

    def merge(self):
        if self.active_adapter not in self.ia3_l.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return

        self.weight = transpose(self.weight, self.fan_in_fan_out)
        self.weight.data = torch.mul(self.weight.data, self.ia3_l[self.active_adapter].data)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.ia3_l.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        self.weight = transpose(self.weight, self.fan_in_fan_out)
        # divide by (IA)^3 vector. Add tolerace to avoid division by zero
        self.weight.data = torch.div(self.weight.data, self.ia3_l[self.active_adapter].data + 1e-8)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.ia3_l.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif not self.merged:
            if self.is_feedforward:
                x = x.to(self.ia3_l[self.active_adapter].dtype)
                interm = x * self.ia3_l[self.active_adapter].flatten()
                result = F.linear(
                    interm.to(self.weight.dtype),
                    transpose(self.weight, self.fan_in_fan_out),
                    bias=self.bias,
                )
            else:
                result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
                result = result.to(self.ia3_l[self.active_adapter].dtype) * self.ia3_l[self.active_adapter].flatten()
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, IA3Layer):
        # (IA)^3 implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            is_feedforward,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_ia3_weights = kwargs.pop("init_ia3_weights", True)
            self.update_layer(adapter_name, init_ia3_weights)
            self.active_adapter = adapter_name
            self.is_feedforward = is_feedforward

        def forward(self, x: torch.Tensor):
            if self.disable_adapters or (self.active_adapter not in self.ia3_l.keys()):
                return super().forward(x)

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()

            ia3_scaling = self.ia3_l[self.active_adapter].flatten()
            if self.is_feedforward:
                result = super().forward(x * ia3_scaling)
            else:
                result = super().forward(x)
                expected_dtype = result.dtype
                result = result * ia3_scaling

                if requires_conversion:
                    result = result.to(expected_dtype)
            return result
