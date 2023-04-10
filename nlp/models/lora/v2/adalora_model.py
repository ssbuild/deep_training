# -*- coding: utf-8 -*-
# @Time:  10:45
# @Author: tk
# @File：adalora_model
import re
import warnings
import torch
from torch import nn
from transformers import Conv1D

from ....layers.lora_v2.layers import mark_only_lora_as_trainable, is_bnb_available, LoraLayer
from ....layers.lora_v2.adalora import RankAllocator,SVDLinear
from ....layers.lora_v2.utils import _freeze_adapter, _get_submodules, \
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
from .lora_model import LoraModel

__all__ = [
    'is_bnb_available',
    'AdaLoraModel'
]



if is_bnb_available():
    import bitsandbytes as bnb
    from ....layers.lora_v2.adalora import SVDLinear8bitLt


class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.


    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            config = self._prepare_adalora_config(config, self.model.config.to_dict())
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "AdaLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.init_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = SVDLinear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = SVDLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        # Calculate the orthogonal regularization
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight
        assert orth_reg_weight > 0

        if hasattr(outputs, "loss"):
            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro")
            regu_loss = regu_loss / num_param
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        lora_config = self.peft_config[adapter_name]
        for name, rank_idx in rank_pattern.items():
            if isinstance(rank_idx, list):
                rank = sum(rank_idx)
            elif isinstance(rank_idx, torch.Tensor):
                rank_idx = rank_idx.view(-1)
                rank = rank_idx.sum().item()
            else:
                raise ValueError("Unexcepted type of rank_idx")
            key = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            _, target, _ = _get_submodules(self.model, key)
            lora_E_weights = target.lora_E[adapter_name][rank_idx]
            lora_A_weights = target.lora_A[adapter_name][rank_idx]
            lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
            ranknum = target.ranknum[adapter_name]
            target.update_layer(
                adapter_name,
                rank,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
            with torch.no_grad():
                if rank > 0:
                    target.lora_E[adapter_name].copy_(lora_E_weights)
                    target.lora_A[adapter_name].copy_(lora_A_weights)
                    target.lora_B[adapter_name].copy_(lora_B_weights)
                    # The scaling is exactly as the previous
                    target.ranknum[adapter_name].copy_(ranknum)

    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
        for name, rank_idx in rank_pattern.items():
            rank = sum(rank_idx)
            prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            for layer in ["lora_E", "lora_A", "lora_B"]:
                key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                if layer != "lora_B":
                    state_dict[key] = (
                        state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
                    )
                else:
                    state_dict[key] = (
                        state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
                    )
        return state_dict

    def update_and_allocate(self, global_step):
        lora_config = self.peft_config[self.trainable_adapter_name]
        # Update the importance score and allocate the budget
        if global_step < lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step)
            if rank_pattern:
                lora_config.rank_pattern = rank_pattern
        # Finalize the budget allocation
        elif global_step == lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, force_mask=True)
            # for some reason, this freezes the trainable parameters and nothing gets updates
            # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
        else:
            return None

    @staticmethod
    def _prepare_adalora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config
