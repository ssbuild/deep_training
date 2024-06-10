# -*- coding: utf-8 -*-
# @Time:  0:28
# @Author: tk
# @File：generator_llava
from typing import List, Tuple
import torch
from transformers import PreTrainedModel, BatchEncoding, BatchFeature
from .generator_base import GeneratorBase
from PIL import Image

class Generate(GeneratorBase):
    def preprocess_inputs(self,query,history = None,**kwargs):
        if history is None:
            history = []
        if not history:
            prompt = query
        else:
            prompt = ''
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt,history

    def build_tokens(self, prompt, max_new_tokens=0,image=None, **kwargs):
        length = self.model_max_length - max_new_tokens
        assert length > 0
        inputs = self.tokenizer([ prompt ], return_tensors="pt")
        for k in inputs:
            if inputs[ k ].size(-1) > length:
                inputs[ k ] = inputs[ k ][ :, :length ]
        # inputs = inputs.to(self.model.device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)} if isinstance(inputs, (
            dict, BatchEncoding)) else inputs.to(self.model.device)

        if isinstance(inputs, (dict, BatchEncoding)):
            inputs = BatchFeature(data={**inputs, "pixel_values": image})

        else:
            inputs = BatchFeature(data={
                "input_ids": inputs.to(self.model.device),
                "pixel_values": image
            })
        return inputs

    def post_process(self, outputs, prompt_length, output_scores=False):
        if output_scores:
            score = outputs.scores[ 0 ]
            return score
        outputs = outputs.tolist()[ 0 ][ prompt_length: ]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return response

    @torch.no_grad()
    def generate(self, query: str,image=None, **kwargs):
        prompt, _ = self.preprocess_inputs(query)
        if image is not None:
            if isinstance(image,str):
                image = Image.open(image)
            image = self.image_processor(image).pixel_values[0]
        inputs = self.build_tokens(prompt, image=image)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs[ 'return_dict_in_generate' ] = True

        outputs = self.model.generate(**inputs, **kwargs)
        prompt_length = 0
        if not self.model.config.is_encoder_decoder:
            prompt_length = len(inputs[ "input_ids" ][ 0 ]) if isinstance(inputs, dict) else len(inputs[ 0 ])
        response = self.post_process(outputs, prompt_length, output_scores)
        return response

    @torch.no_grad()
    def chat(self, query: str, history: List[ Tuple[ str, str ] ] = None,image=None, **kwargs):
        prompt, history = self.preprocess_inputs(query, history)
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            image = self.image_processor(image).pixel_values[ 0 ]
        inputs = self.build_tokens(prompt, image=image)
        output_scores = kwargs.get('output_scores', False)
        if output_scores:
            kwargs[ 'return_dict_in_generate' ] = True

        outputs = self.model.generate(**inputs, **kwargs)
        prompt_length = 0
        if not self.model.config.is_encoder_decoder:
            prompt_length = len(inputs[ "input_ids" ][ 0 ]) if isinstance(inputs, (dict, BatchEncoding)) else len(
                inputs[ 0 ])
        response = self.post_process(outputs, prompt_length, output_scores)
        return response, history