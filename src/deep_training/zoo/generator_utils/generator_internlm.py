# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import List, Tuple

import torch
from transformers import BatchEncoding

from .generator_base import GeneratorBase

class Generate(GeneratorBase):

    def preprocess_inputs(self, query: str, history=None,**kwargs):
        if history is None:
            history = []
        prompt = ""
        for record in history:
            prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return prompt,history

    def post_process(self,outputs,prompt_length,output_scores=False):
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<eoa>")[0]
        return response

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None,  eos_token_id = (2, 103028),**kwargs):
        prompt, history = self.preprocess_inputs(query, history)
        inputs = self.build_tokens(prompt)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True
        outputs = self.model.generate(**inputs,eos_token_id=eos_token_id, **kwargs)
        prompt_length = len(inputs["input_ids"][0]) if isinstance(inputs,(dict,BatchEncoding)) else len(inputs[0])
        response = self.post_process(outputs, prompt_length, output_scores)
        return response, history

