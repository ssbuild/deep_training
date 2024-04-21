# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/29 9:46
import torch
from torch import nn

from .llm_model import TransformerForLM
from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *

import logging
logger = logging.getLogger(__name__)

class RewardModel(TransformerForLM):
    def __init__(self, *args, **kwargs):
        super(RewardModel, self).__init__(*args, **kwargs)

        base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith('_') else self.base_model_prefix
        self.model_key = base_model_prefix
        transformer_bone = getattr(self.model,base_model_prefix,None)
        assert transformer_bone is not None
        hidden_size = self.config.word_embed_proj_dim if getattr(self.config,'word_embed_proj_dim',None) else self.config.hidden_size
        self.score = nn.Linear(hidden_size, self.config.num_labels)
        self.pad_token_id = self.config.pad_token_id or self.config.eos_token_id

    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def forward_value(self,**batch):
        state = getattr(self.model,self.model_key,None)(**batch)[0]
        value = self.score(state)
        return value.squeeze(-1)


    def forward_loss(self,
                     chosen_ids: torch.Tensor, chosen_values: torch.Tensor,
                     rejected_ids: torch.Tensor, rejected_values: torch.Tensor):

        pad_token_id = self.pad_token_id
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        bs = chosen_ids.size(0)
        # pad_id = torch.tensor(self.config.pad_token_id, dtype=chosen_ids.dtype, device=chosen_values.device)
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_value = chosen_values[i]
            rejected_value = rejected_values[i]

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_id == pad_token_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_id.shape[0]
            r_inds = (rejected_id == pad_token_id).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_id.shape[0]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_id != rejected_id).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_value[divergence_ind:end_ind]
            r_truncated_reward = rejected_value[divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_mean_scores.append(c_truncated_reward[-1])
            rejected_mean_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        loss /= bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return loss,chosen_mean_scores,rejected_mean_scores

    def forward_score(self,input_ids,values):
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_mean_scores = [
        ]  # we use this name for consistency with the original forwad function
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            c_inds = (input_id == self.pad_token_id).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            chosen_mean_scores.append(value[c_ind - 1])
        return torch.stack(chosen_mean_scores)

    def forward_returns(self, **inputs):
        input_ids = inputs['input_ids']
        values = self.forward_value(**inputs)
        ends = torch.argmax((input_ids == self.config.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(values, 1, ends).squeeze(-1)
        return returns

    def compute_loss(self, *args, return_value_only=False, **batch) -> tuple:
        input_a, input_b = {}, {}
        for k, v in batch.items():
            i, k = (input_b, k[:-1]) if k.endswith('2') else (input_a, k)
            i[k] = v

        value_a = self.forward_value(**input_a)
        if len(input_b) > 0:
            value_b = self.forward_value(**input_b)
            loss, chosen_mean_scores, rejected_mean_scores = self.forward_loss(input_a["input_ids"], value_a,
                                                                               input_b["input_ids"], value_b)
            loss_dict = {
                "loss": loss,
                "chosen_mean_scores": chosen_mean_scores.mean(),
                "rejected_mean_scores": rejected_mean_scores.mean()
            }
            if self.training:
                return (loss_dict,)
            return (loss, value_a, value_b,chosen_mean_scores,rejected_mean_scores)


        if return_value_only:
            return (value_a,)
        scores = self.forward_score(batch["input_ids"], value_a)
        return (value_a, scores)




class MyRewardTransformer(RewardModel, ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,**kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyRewardTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.inject_model()

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.enable:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.enable:
            return [(self.backbone, lr)]
        return super(MyRewardTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model

    def forward_returns(self,*args,**kwargs):
        if self.lora_args is not None and self.lora_args.enable:
            model = self.backbone.model
        else:
            model = self.backbone
        return model.forward_returns(*args,**kwargs)