# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 14:25


import copy
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable, Dict, Union
import torch
from torch import nn
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.chatglm import ChatGLMForConditionalGeneration,ChatGLMConfig,setup_model_profile # noqa
from deep_training.nlp.models.transformer import TransformerBase
from deep_training.nlp.losses.loss_dpo import dpo_loss
from .generation_utils import build_masks_and_position_ids_glm
from .tokenization_chatglm import ChatGLMTokenizer
from ..auto.base_wapper import BaseModelWrapper
from ...utils.dpo_utils import DpoModule
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)

class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self,config):
        super(MyChatGLMForConditionalGeneration, self).__init__(config)


class TransformerDPOForLM(DpoModule,TransformerBase):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False,**kwargs):
        super(TransformerDPOForLM, self).__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))
        self.beta = beta
        self.ref_free = ref_free
        self.ref_model = ref_model

        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        # class CastOutputToFloat(nn.Sequential):
        #     def forward(self, x):
        #         return super().forward(x).to(torch.float32)
        #
        # self.model.lm_head = CastOutputToFloat(self.model.lm_head)


    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
        self.model.enable_input_require_grads()




class TransformerDPO(TransformerDPOForLM,ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: PetlArguments = kwargs.pop('lora_args',None)
        num_layers_freeze = kwargs.pop('num_layers_freeze',-1)
        super(TransformerDPO, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = None
        self.num_layers_freeze = num_layers_freeze
        #可能添加新词
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.enable:
            return [(self.backbone, lr)]
        return super(TransformerDPO, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        return self.backbone.model
