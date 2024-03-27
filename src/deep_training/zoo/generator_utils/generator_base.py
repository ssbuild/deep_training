# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:27
from typing import List, Tuple
import torch
from transformers import PreTrainedModel, BatchEncoding


class GeneratorBase:
    def __init__(self,model : PreTrainedModel,tokenizer,**kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.config = kwargs.get('config',None) or self.model.config
        self.generation_config = kwargs.get('generation_config',None) or getattr(self.model,'generation_config',None)
        self.model_max_length = kwargs.get('model_max_length',None) or getattr(self.config,'model_max_length',None) or 65535
        self.kwargs = kwargs


    def preprocess_inputs(self,query,history = None,**kwargs):
        return query,history or []

    def build_tokens(self,prompt,max_new_tokens=0):
        length = self.model_max_length - max_new_tokens
        assert length > 0
        inputs = self.tokenizer([prompt], return_tensors="pt")
        for k in inputs:
            if inputs[k].size(-1) > length:
                inputs[k] = inputs[k][:,:length]
        # inputs = inputs.to(self.model.device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)} if isinstance(inputs,(dict,BatchEncoding)) else inputs.to(self.model.device)
        return inputs

    def post_process(self,outputs,prompt_length,output_scores=False):
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return response

    @torch.no_grad()
    def generate(self, query: str, **kwargs):
        prompt,_ = self.preprocess_inputs(query)
        inputs = self.build_tokens(prompt)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True

        outputs = self.model.generate(**inputs, **kwargs)
        prompt_length = 0
        if not self.model.config.is_encoder_decoder:
            prompt_length = len(inputs["input_ids"][0]) if isinstance(inputs,dict) else len(inputs[0])
        response = self.post_process(outputs, prompt_length,output_scores)
        return response


    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None,  **kwargs):
        prompt, history = self.preprocess_inputs(query,history)
        inputs = self.build_tokens(prompt)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs['return_dict_in_generate'] = True

        outputs = self.model.generate(**inputs, **kwargs)
        prompt_length = 0
        if not self.model.config.is_encoder_decoder:
            prompt_length = len(inputs["input_ids"][0]) if isinstance(inputs,(dict,BatchEncoding)) else len(inputs[0])
        response = self.post_process(outputs, prompt_length,output_scores)
        return response,history

    @torch.no_grad()
    def chat_stream(self, query: str, history: List[Tuple[str, str]] = None, **kwargs):
        raise NotImplemented