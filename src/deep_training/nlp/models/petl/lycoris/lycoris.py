# -*- coding: utf-8 -*-
# @Time:  23:54
# @Author: tk
# @File：lycoris_model
import re
import warnings
from itertools import chain
from typing import Dict, Type, Union
import torch
from torch import nn
from tqdm import tqdm
from ..petl_model_base import PetlModelBase
from ..config.lycoris import LycorisConfig
from ....layers.petl.lycoris.layer import LycorisLayer
from ....layers.petl.petl_layer import check_target_module_exists, PetlLayerBase
from ....layers.petl.utils import ModulesToSaveWrapper, _get_submodules


class PetlLycorisBase(PetlModelBase):
    r"""
    A base tuner for LyCORIS like adapters
    """

    prefix: str
    layers_mapping: Dict[Type[torch.nn.Module], Type[LycorisLayer]]

    def __init__(self, model, config, adapter_name,**kwargs):
        super().__init__(model, config, adapter_name,**kwargs)

    @staticmethod
    def _check_target_module_exists(config, key):
        return check_target_module_exists(config, key)

    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[LycorisLayer, nn.Module],
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(config.rank_pattern.keys(), config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f"(.*\.)?{key}$", current_key), pattern_keys), target_name)

        kwargs = config.to_dict()
        kwargs["r"] = config.rank_pattern.get(target_name_key, config.r)
        kwargs["alpha"] = config.alpha_pattern.get(target_name_key, config.alpha)

        if isinstance(target, LycorisLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @classmethod
    def _create_new_module(cls, config: LycorisConfig, adapter_name: str, target: nn.Module, **kwargs) -> LycorisLayer:
        # Find corresponding subtype of provided target module
        new_module_cls = None
        for subtype, target_cls in cls.layers_mapping.items():
            if isinstance(target, subtype):
                new_module_cls = target_cls
                break

        # We didn't find corresponding type, so adapter for this layer is not supported
        if new_module_cls is None:
            raise ValueError(
                f"Target module not found, currently only adapters for {', '.join([x.__name__ for x in cls.modules_mapping.keys()])} are supported"
            )

        if isinstance(target, torch.nn.Conv2d):
            new_module = new_module_cls(
                target.in_channels,
                target.out_channels,
                target.weight.size()[2:],
                stride=target.stride,
                padding=target.padding,
                dilation=target.dilation,
                groups=target.groups,
                bias=target.bias is not None,
                padding_mode=target.padding_mode,
                device=target.weight.device,
                dtype=target.weight.dtype,
                adapter_name=adapter_name,
                **kwargs,
            )
        elif isinstance(target, torch.nn.Linear):
            new_module = new_module_cls(
                target.in_features,
                target.out_features,
                bias=target.bias is not None,
                device=target.weight.device,
                dtype=target.weight.dtype,
                adapter_name=adapter_name,
                **kwargs,
            )
        else:
            raise ValueError(
                "Target module not found, currently only adapters for nn.Linear and nn.Conv2d are supported"
            )

        return new_module

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _prepare_adapter_config(petl_config, model_config):
        if petl_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `petl_config`")
        return petl_config

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (PetlLayerBase, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LOHA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if "hada" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LycorisLayer):
                if isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                elif isinstance(target, nn.Linear):
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        device=target.weight.device,
                    )
                else:
                    raise ValueError(
                        "Cannot convert current module to torch module, currently only adapters for nn.Linear and nn.Conv2d are supported"
                    )
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def merge_and_unload(self, progressbar: bool = False):
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LycorisLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (`str`): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.petl_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.petl_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LycorisLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if hasattr(self.model, name):
                return getattr(self.model, name)
            return getattr(self.model.model, name)