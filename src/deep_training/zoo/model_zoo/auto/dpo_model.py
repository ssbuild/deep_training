# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
from typing import Any

import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.transformer import TransformerForCausalLM
from .base_wapper import BaseModelWrapper
from ...utils.dpo_utils import DpoModule
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *

import logging
logger = logging.getLogger(__name__)



class TransformerDPOForLM(DpoModule,TransformerForCausalLM):
    def __init__(self, *args,ref_model=None,beta=0.1,ref_free=False, **kwargs):
        super(TransformerDPOForLM, self).__init__(*args, **kwargs)
        self.beta=beta
        self.ref_free=ref_free
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
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()


class MyTransformerDPO(BaseModelWrapper,TransformerDPOForLM, ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformerDPO, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.rope_args = rope_args
        #可能扩充词表
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()
