# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/7 16:08
from torch import nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM,logger
from .configuration_llama import configuration_llama # noqa
from ..transformer_base import TransformerBase
from ...utils.torch_utils import skip_init # noqa


class ReLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.quantized = False
        if hasattr(self.config,"quantization_bit") and self.config.quantization_bit in [4, 8]:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def quantize(self, bits: int, empty_init=False, device=None, **kwarg):
        if bits == 0:
            return
        from .quantization import quantize
        if self.quantized:
            logger.info("Already quantized.")
            return self
        quantize(self, bits=bits, empty_init=empty_init, device=device, **kwarg)
        self.config.quantization_bit = bits
        self.quantized = True
        return self

class TransformerLlamaLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerLlamaLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(ReLlamaForCausalLM, *args, **kwargs))
