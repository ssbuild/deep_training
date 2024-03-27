# -*- coding: utf-8 -*-
# @Time:  23:49
# @Author: tk
# @File：LAYER
import math
import warnings
from abc import abstractmethod
from typing import Union, Tuple, Optional, Set
import torch
from torch import nn
from torch.nn import functional as F

from ..petl_layer import PetlLayerBase
from ..utils import transpose, is_bnb_available,is_bnb_4bit_available,is_auto_gptq_available,is_optimum_available

if is_bnb_available():
    import bitsandbytes as bnb


class LycorisLayer(PetlLayerBase, nn.Module):
    r"""
    A base layer for LyCORIS like adapters
    """
    # adapter_layer_names needs to be defined on the child class
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self):
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}

        # Tuner info
        self._disable_adapters = False
        self.merged_adapters = []

    @property
    @abstractmethod
    def _available_adapters(self) -> Set[str]:
        ...

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def _op(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._op(x, self.weight)
        elif self.merged:
            result = self._op(x, self.weight)
        else:
            # Get base weights
            weight = self.weight.data

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    weight = weight + self.get_delta_weight(active_adapter)

            # Perform actual operation
            result = self._op(x, weight)

        result = result.to(previous_dtype)
        return result

    @abstractmethod
    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        ...

    def merge(self) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self._available_adapters:
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    @abstractmethod
    def reset_adapter_parameters(self, adapter_name: str):
        ...

    def set_scale(self, adapter, scale):
        if adapter not in self._available_adapters:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue

            self.scaling[active_adapter] *= scale

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.weight.data -= self.get_delta_weight(active_adapter)

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue

            if scale is None:
                self.scaling[active_adapter] = self.alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    @abstractmethod
    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
        ...
