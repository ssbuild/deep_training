# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/20 10:12
from typing import Optional, Any
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.transformer import TransformerForCausalLM
from .base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *


import logging
logger = logging.getLogger(__name__)




class TransformerForLM(TransformerForCausalLM):
    def __init__(self, *args, **kwargs):
        super(TransformerForLM, self).__init__(*args, **kwargs)
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

class MyTransformer(BaseModelWrapper,TransformerForLM, ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,
                 lora_args: Optional[LoraConfig] = None,
                 prompt_args: Optional[PromptLearningConfig] = None,
                 new_num_tokens=None,rope_args=None, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.rope_args = rope_args

        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))   #可能扩充词表
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()

