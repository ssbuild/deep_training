# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 9:07
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from ..petl_layer import PetlLayerAbstract
from ..utils import is_bnb_available, transpose

if is_bnb_available():
    import bitsandbytes as bnb

class IA3Layer(PetlLayerAbstract):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
    ):
        self.scaling = {}
        self.ia3_l = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.is_feedforward = is_feedforward

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l.update(nn.ParameterDict({adapter_name: nn.Parameter(weight)}))
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)




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
