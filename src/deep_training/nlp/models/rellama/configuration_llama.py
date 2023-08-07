# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/7 16:06

from transformers.models.llama.configuration_llama import * # noqa
from transformers.models.llama.configuration_llama import LlamaConfig as LlamaConfig_

class LlamaConfig(LlamaConfig_):
    def __init__(self,
                 quantization_bit=0,
                 initializer_weight=False,
                 **kwargs,):
        super().__init__(**kwargs)
        self.quantization_bit = quantization_bit
        self.initializer_weight = initializer_weight
