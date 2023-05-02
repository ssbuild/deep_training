import json
import os
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, List, Literal, AnyStr




import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Union, List

import torch
from transformers import Conv1D
from transformers.utils import PushToHubMixin

from .configuration import LoraArguments, WEIGHTS_NAME
from ....layers.lora_v1.layers import MergedLinear, is_bnb_available, LoraLayer, Linear
from ....layers.lora_v1.utils import mark_only_lora_as_trainable

__all__ = [
    'LoraArguments',
    'LoraModel',
    'LoraLayer'
]

if is_bnb_available():
    import bitsandbytes as bnb
    from ....layers.lora_v1.layers import Linear8bitLt


def get_lora_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Lora model.

    Args:
        model ([`LoraModel`]): The Lora model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()

    # to_return = lora_state_dict(model, bias=model.lora_config.bias)
    # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
    # to directly with the state dict which is necessary when using DeepSpeed or FSDP
    bias = model.lora_config.bias
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def set_lora_model_state_dict(model, lora_model_state_dict):
    """
    Set the state dict of the Lora model.

    Args:
        model ([`LoraModel`]): The Lora model.
        lora_model_state_dict (`dict`): The state dict of the Lora model.
    """

    model.load_state_dict(lora_model_state_dict, strict=False)
    return model



class LoraModel(torch.nn.Module,PushToHubMixin):
    def __init__(self, model, config):
        torch.nn.Module.__init__(self)
        PushToHubMixin.__init__(self)
        self.lora_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.lora_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if not loaded_in_8bit:
            if hasattr(self.model, 'model'):
                loaded_in_8bit = getattr(self.model.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "lora_dropout": self.lora_config.lora_dropout,
            "fan_in_fan_out": self.lora_config.fan_in_fan_out,
            "merge_weights": self.lora_config.merge_weights or self.lora_config.inference_mode,
        }

        if self.lora_config.target_dtype is not None and not loaded_in_8bit:
            if self.lora_config.target_dtype == 16 or self.lora_config.target_dtype == '16':
                kwargs['dtype'] = torch.float16
            elif self.lora_config.target_dtype == 32 or self.lora_config.target_dtype == '32':
                kwargs['dtype'] = torch.float32
            elif self.lora_config.target_dtype == 64 or self.lora_config.target_dtype == '64':
                kwargs['dtype'] = torch.float64
            elif self.lora_config.target_dtype == 'bf16':
                kwargs['dtype'] = torch.bfloat16
            elif isinstance(self.lora_config.target_dtype,torch.dtype):
                kwargs['dtype'] = self.lora_config.target_dtype

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.lora_config.target_modules, str):
                target_module_found = re.fullmatch(self.lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt) and self.lora_config.enable_lora is None:
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.lora_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.lora_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.lora_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = target.weight.shape
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            try:
                return getattr(self.model, name)
            except AttributeError:
                return getattr(self.model.model, name)
    @property
    def modules_to_save(self):
        return None

    def get_lora_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.lora_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_lora_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        inference_mode = self.lora_config.inference_mode
        self.lora_config.inference_mode = True
        self.lora_config.save_pretrained(save_directory)
        self.lora_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, pretrained_model_name_or_path,lora_config: LoraArguments = None, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """
        if lora_config is None:
            lora_config: LoraArguments = LoraArguments.from_pretrained(pretrained_model_name_or_path)

        model = cls(model, lora_config)
        # load weights if any
        if os.path.exists(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
            filename = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            raise ValueError(
                f"Can't find weights for {pretrained_model_name_or_path} in {pretrained_model_name_or_path} or in the Hugging Face Hub. "
                f"Please check that the file {WEIGHTS_NAME} is present at {pretrained_model_name_or_path}."
            )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # load the weights into the model
        model = set_lora_model_state_dict(model, adapters_weights)
        return model

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)