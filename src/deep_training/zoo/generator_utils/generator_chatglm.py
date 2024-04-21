# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import List, Tuple
import torch
from transformers import LogitsProcessorList
from aigc_zoo.model_zoo.chatglm.llm_model import InvalidScoreLogitsProcessor
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

    @torch.no_grad()
    def chat_stream(self, query: str, history: List[Tuple[str, str]] = None, **kwargs):
        return self.model.stream_chat(self.tokenizer, query=query, history=history, **kwargs)

    # @torch.no_grad()
    # def chat_stream(self, query: str, history: List[Tuple[str, str]] = None,logits_processor=None, **kwargs):
    #     if history is None:
    #         history = []
    #     if logits_processor is None:
    #         logits_processor = LogitsProcessorList()
    #     logits_processor.append(InvalidScoreLogitsProcessor())
    #     gen_kwargs = {"logits_processor": logits_processor, ** kwargs}
    #
    #     tokenizer = self.tokenizer
    #     prompt,history = self.preprocess_inputs(query,history)
    #     inputs = self.build_tokens(prompt)
    #
    #     for outputs in self.model.stream_generate(**inputs, **gen_kwargs):
    #         outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    #         response = tokenizer.decode(outputs)
    #         response = self.model.process_response(response)
    #         new_history = history + [(query, response)]
    #         yield response, new_history
