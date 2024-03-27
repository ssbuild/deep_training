# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import List, Tuple

import torch
from transformers import LogitsProcessorList

from .generator_base import GeneratorBase
from ..model_zoo.chatglm2.llm_model import InvalidScoreLogitsProcessor


class Generate(GeneratorBase):
    def preprocess_inputs(self,query,history = None,**kwargs):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt,history

    @torch.no_grad()
    def chat_stream(self, query: str, history: List[Tuple[str, str]] = None, **kwargs):
        return self.model.stream_chat(self.tokenizer, query=query, history=history ,**kwargs)