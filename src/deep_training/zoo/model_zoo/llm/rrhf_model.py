# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 16:41
import torch
from deep_training.nlp.models.transformer import TransformerForCausalLM
from torch.nn import functional as F

from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)

__all__ = [
    'RRHFModelForCausalLM',
    'MyRRHFTransformer',
]

class RRHFModelForCausalLM(TransformerForCausalLM):
    def __init__(self, *args, length_penalty=1.0, rrhf_weight=1.0, **kwargs):
        super(RRHFModelForCausalLM, self).__init__(*args, **kwargs)
        self.length_penalty = length_penalty
        self.rrhf_weight = rrhf_weight


    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def gather_logits_labels(self, logits, labels,mask):
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, mask):
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.length_penalty)
        return scores

    def rrhf_loss(self, scores, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, **inputs):
        labels = inputs.pop('labels',None)
        scores = inputs.pop('scores', None)
        logits = self.model(**inputs)[0]  # (batch * cand) * L * V
        if labels is not None:
            labels = labels.long()
            logits = F.log_softmax(logits, dim=-1)
            mask = (labels != -100).float()
            logit_label = self.gather_logits_labels(logits, labels,mask)
            compute_scores = self.get_score(logit_label,mask)
            rrhf_loss = self.rrhf_loss(compute_scores, scores)
            sft_loss = self.sft_loss(logit_label, scores)
            loss = self.rrhf_weight * rrhf_loss + sft_loss
            loss_dict = {
                "rrhf_loss": rrhf_loss,
                "sft_loss": sft_loss,
                "loss": loss
            }
            return (loss_dict,)
        return (logits,)







class MyRRHFTransformer(RRHFModelForCausalLM,ModelWeightMixin,BaseModelWrapper,with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyRRHFTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.prompt_args=prompt_args

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
        return super(MyRRHFTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.get_llm_model().generate(*args, **kwargs)