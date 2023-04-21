# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 11:32
""" PyTorch Moss model."""
from ..transformer_base import TransformerBase
from .modeling_moss import MossPreTrainedModel,MossConfig,MossModel,MossForCausalLM



class TransformerMossForCausalLM(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerMossForCausalLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MossForCausalLM, *args, **kwargs))