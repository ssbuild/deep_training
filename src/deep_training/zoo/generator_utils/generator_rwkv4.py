# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
import torch
from transformers import PreTrainedTokenizer
from .generator_base import GeneratorBase

class Generate(GeneratorBase):
    def preprocess_inputs(self,query,history = None,**kwargs):
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

    def build_token_ids(self,prompt):
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(self.model.device)
        else:
            inputs = self.tokenizer.encode(prompt)
            inputs = torch.tensor([inputs], dtype=torch.int64)
            inputs = inputs.to(self.model.device)
        return inputs

    def post_process(self, outputs, prompt_length, output_scores=False):
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][prompt_length:]
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            response = self.tokenizer.decode(outputs,skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(outputs)
        return response

