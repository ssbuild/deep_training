# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 14:47

import re
from deep_training.nlp.layers.rope_scale.patch import *
from typing import List, Tuple
import torch
from torch import nn
from deep_training.nlp.models.moss import MossForCausalLM,MossConfig # noqa
from deep_training.nlp.models.moss.tokenization_moss import MossTokenizer # noqa
from deep_training.nlp.models.transformer import TransformerBase

from ..auto.base_wapper import BaseModelWrapper
from ...utils.dpo_utils import DpoModule
from ...weight.modelweighter import *
from ...utils.transformer_utils import hf_decorator
import logging
logger = logging.getLogger(__name__)

class MyMossForCausalLM(MossForCausalLM):
    def __init__(self,config):
        super(MyMossForCausalLM, self).__init__(config)
        # self.transformer.gradient_checkpointing = True


class TransformerDPOForLM(DpoModule,TransformerBase):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False, **kwargs):
        super(TransformerDPOForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyMossForCausalLM, *args, **kwargs))
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
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()




class TransformerDPO(TransformerDPOForLM,ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args',None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        num_layers_freeze = kwargs.pop('num_layers_freeze', -1)
        super(TransformerDPO, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.num_layers_freeze = num_layers_freeze

        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        # 可能扩充词表
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
        return super(TransformerDPO, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyMossForCausalLM:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model
