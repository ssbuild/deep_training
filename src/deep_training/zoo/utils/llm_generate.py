# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/16 16:37

from typing import List, Tuple
import torch
from transformers import PreTrainedModel,PreTrainedTokenizer

class Generate:
    @classmethod
    def build_inputs(cls,query,history = None):
        if history is None:
            history = []
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt,history
    @classmethod
    @torch.no_grad()
    def generate(cls,model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str, **kwargs):
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True
        prompt = query
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **kwargs)
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs,skip_special_tokens=True)
        return response

    @classmethod
    @torch.no_grad()
    def chat(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str, history: List[Tuple[str, str]] = None,  **kwargs):
        prompt,history = Generate.build_inputs(query,history)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **kwargs)
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs,skip_special_tokens=True)
        history = history + [(query, response)]
        return response, history