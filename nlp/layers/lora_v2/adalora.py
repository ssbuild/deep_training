# @Time    : 2023/4/7 23:25
# @Author  : tk
# @FileName: adalora
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from .layers import LoraLayer,is_bnb_available
from .utils import transpose

if is_bnb_available():
    import bitsandbytes as bnb

class AdaLoraLayer(LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__(in_features, out_features)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights,dtype=None):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # Right singular vectors
        self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(r, self.in_features,dtype=dtype))}))
        # Singular values
        self.lora_E.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(r, 1,dtype=dtype))}))
        # Left singular vectors
        self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(self.out_features, r,dtype=dtype))}))
        # The current rank
        self.ranknum.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(1,dtype=dtype), requires_grad=False)}))
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class SVDLinear(nn.Linear, AdaLoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter]
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]),
                    self.fan_in_fan_out
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter]
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]),
                    self.fan_in_fan_out
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            result += (
                (
                    self.lora_dropout[self.active_adapter](x)
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                    @ self.lora_B[self.active_adapter].T
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result


if is_bnb_available():

    class SVDLinear8bitLt(bnb.nn.Linear8bitLt, AdaLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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
            AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights,dtype=kwargs.get('dtype',None))
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        (
                            self.lora_dropout[self.active_adapter](x)
                            @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                            @ self.lora_B[self.active_adapter].T
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                        / (self.ranknum[self.active_adapter] + 1e-5)
                    )
                else:
                    output = (
                        (
                            self.lora_dropout[self.active_adapter](x)
                            @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                            @ self.lora_B[self.active_adapter].T
                        )
                        * self.scaling[self.active_adapter]
                        / (self.ranknum[self.active_adapter] + 1e-5)
                    )
                result += output
            return result


class RankAllocator(object):
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)
