# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:14
from .generator_base import GeneratorBase

class Generate(GeneratorBase):
    def preprocess_inputs(self,query,history = None,**kwargs):
        if history is None:
            history = []
        prompt = ""
        if history is not None:
            for q, a in history:
                prompt += "用户：{}\n小元：{}".format(q, a)
        prompt += "用户：{}\n小元：".format(query)
        return prompt,history


