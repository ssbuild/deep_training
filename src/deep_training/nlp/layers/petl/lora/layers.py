# @Time    : 2023/4/10 23:58
# @Author  : tk
# @FileName: layers
import math
import warnings
from typing import Union, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

from ..petl_layer import PetlLayerAbstract
from ..utils import transpose, is_bnb_available,is_bnb_4bit_available,is_auto_gptq_available,is_optimum_available

if is_bnb_available():
    import bitsandbytes as bnb


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer(PetlLayerAbstract):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"]

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

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

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
            self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features))
            weight_B = torch.randn((self.out_features, r))
            self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
            self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(self.weight.device, dtype=weight.dtype)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def scale_layer(self, scale_factor: float) -> None:
        if scale_factor != 1:
            for active_adapter in self.active_adapters:
                alpha = self.lora_alpha[active_adapter]
                r = self.r[active_adapter]
                self.scaling[active_adapter] = (alpha / r) * scale_factor

    def unscale_layer(self) -> None:
        for active_adapter in self.active_adapters:
            alpha = self.lora_alpha[active_adapter]
            r = self.r[active_adapter]
            self.scaling[active_adapter] = alpha / r





class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)

    def merge(self) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)
                self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[adapter]
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            result = self._linear(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result



class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Embedding, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_embedding_A.keys():
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)
                self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return transpose(self.lora_embedding_B[adapter] @ self.lora_embedding_A[adapter], True) * self.scaling[adapter]

    def _embed(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        return F.embedding(
            input,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._embed(x)
        elif self.merged:
            result = self._embed(x)
        else:
            result = self._embed(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result += (after_A @ embedding_B) * scaling

        return result


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Conv2d, in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)
                self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            return (
                self.lora_B[adapter].weight.squeeze(3).squeeze(2) @ self.lora_A[adapter].weight.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            # conv2d 3x3
            return (
                F.conv2d(
                    self.lora_A[adapter].weight.permute(1, 0, 2, 3),
                    self.lora_B[adapter].weight,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

    def _conv2d(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif self.merged:
            result = self._conv2d(x)
        else:
            result = self._conv2d(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result




if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ) -> None:
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
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.set_adapter(adapter_name)

        def merge(self):
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                if self.state.SCB is None:
                    self.state.SCB = self.weight.SCB
                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = torch.eye(self.weight.data.shape[-1]).contiguous().half().to(self.weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.state.CxB is None:
                    self.state.CxB, self.state.SB = bnb.functional.transform(
                        self.weight.data, to_order=self.state.formatB
                    )
                out32, Sout32 = bnb.functional.igemmlt(im, self.state.CxB, Sim, self.state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, self.state.SCB, bias=None).t()
                w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                self.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged_adapters.append(active_adapter)
                self.merged = True

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                lora_data = self.get_delta_weight(active_adapter)

                if self.state.SCB is None:
                    self.state.SCB = self.weight.SCB
                im = torch.eye(self.weight.data.shape[-1]).contiguous().half().to(self.weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.state.CxB is None:
                    self.state.CxB, self.state.SB = bnb.functional.transform(
                        self.weight.data, to_order=self.state.formatB
                    )
                out32, Sout32 = bnb.functional.igemmlt(im, self.state.CxB, Sim, self.state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, self.state.SCB, bias=None).t()
                w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                self.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged = False

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = super().forward(x)
            elif self.merged:
                result = super().forward(x)
            else:
                result = super().forward(x)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result


if is_bnb_4bit_available():

    class Linear4bit(bnb.nn.Linear4bit, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.set_adapter(adapter_name)

        def merge(self):
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                kwargs = self.weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)
                w_data = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state) + lora_data
                self.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = True

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                kwargs = self.weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)
                w_data = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state) - lora_data
                self.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = False

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = super().forward(x)
            elif self.merged:
                result = super().forward(x)
            else:
                result = super().forward(x)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = x.to(lora_A.weight.dtype)

                    output = lora_B(lora_A(dropout(x)))
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result




class QuantLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        LoraLayer.__init__(
            self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result += output
        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
