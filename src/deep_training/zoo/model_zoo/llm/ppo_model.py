# coding=utf8
# @Time    : 2023/5/12 21:31
# @Author  : tk
# @FileName: llm_model

from typing import List, Tuple, Optional, Union
import torch
from deep_training.nlp.models.rl.modeling_ppo import AutoModelForCausalLMWithValueHead
from deep_training.nlp.rl.ppo.configuration import PPOConfig,PPOArguments
from deep_training.nlp.rl.ppo.ppo_module import PPOModelLoss
from transformers import AdamW

from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)

class PPOModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    def __init__(self,*args,hidden_size=None, up_sampling_score=False,**kwargs):
        config = kwargs.get('config')
        if hidden_size is None:
            hidden_size = config.word_embed_proj_dim if getattr(config, 'word_embed_proj_dim',
                                                                None) else config.hidden_size
        super(PPOModelForCausalLMWithValueHead, self).__init__(*args,hidden_size=hidden_size, up_sampling_score=up_sampling_score, **kwargs)



    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()








class MyPPOTransformer(PPOModelForCausalLMWithValueHead, PPOModelLoss,ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        ppo_args: PPOConfig = kwargs.pop('ppo_args', None)
        super(MyPPOTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ppo_config = ppo_args
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
        return super(MyPPOTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self,*args,**kwargs):
        return self.get_llm_model().generate(*args,**kwargs)

    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        training_args = self.training_args
        optimizer = AdamW(p, lr=training_args.learning_rate,
                          eps=training_args.adam_epsilon,
                          betas=training_args.optimizer_betas,
                          weight_decay=training_args.weight_decay)
        return optimizer


    def training_step(self,*args, **inputs):
        outputs = self.compute_loss(*args, **inputs)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        return self.forward_ppo_loss(*args, **inputs)


    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)

