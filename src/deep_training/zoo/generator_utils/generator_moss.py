# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import List, Tuple
import torch
from transformers import BatchEncoding

from .generator_base import GeneratorBase

class Generate(GeneratorBase):

    def preprocess_inputs(self,query: str,history: List[Tuple[str, str]] = None,
                          meta_instruction=None,plugin_instruction=None,**kwargs):
        if history is None:
            history = []
        prompt = meta_instruction or self.model.get_meta_instruction()
        if plugin_instruction is not None:
            prompt += plugin_instruction
        for i, (old_query, response) in enumerate(history):
            prompt += "<|Human|>: {}<eoh>\n<|MOSS|>:{}\n".format(old_query, response)
        prompt += "<|Human|>: {}<eoh>\n<|MOSS|>:".format(query)
        return prompt,history

    @torch.no_grad()
    def generate(self, query: str,
                 meta_instruction=None,
                 plugin_instruction=None,
                 **kwargs):
        prompt,_ = self.preprocess_inputs(query,
                                          meta_instruction=meta_instruction,
                                          plugin_instruction=plugin_instruction)
        inputs = self.build_tokens(prompt)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True

        outputs = self.model.generate(**inputs, **kwargs)

        prompt_length = len(inputs["input_ids"][0]) if isinstance(inputs,dict) else len(inputs[0])
        response = self.post_process(outputs, prompt_length,output_scores)
        return response


    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None,
             meta_instruction=None,
             plugin_instruction=None,
             **kwargs):
        prompt, history = self.preprocess_inputs(query,history,
                                                 meta_instruction=meta_instruction,
                                                 plugin_instruction=plugin_instruction)
        inputs = self.build_tokens(prompt)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True
        outputs = self.model.generate(**inputs, **kwargs)
        prompt_length = len(inputs["input_ids"][0]) if isinstance(inputs,(dict,BatchEncoding)) else len(inputs[0])
        response = self.post_process(outputs, prompt_length,output_scores)
        return response,history