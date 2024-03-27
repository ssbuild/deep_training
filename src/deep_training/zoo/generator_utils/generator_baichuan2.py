# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import List, Optional
import torch
from transformers import GenerationConfig, PreTrainedModel
from .generator_base import GeneratorBase

class Generate(GeneratorBase):
    def build_tokens(self, messages: List[dict], max_new_tokens: int=0):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = self.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.config.model_max_length // 2, max_input_tokens)
        total_input, round_input = [], []
        user_token_id = getattr(self.generation_config,'user_token_id',None)
        assistant_token_id = getattr(self.generation_config,'assistant_token_id',None)
        for i, message in enumerate(messages[::-1]):
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [user_token_id] + content_tokens + round_input if user_token_id is not None else content_tokens + round_input
                if total_input and len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message['role'] == 'assistant':
                round_input = [assistant_token_id] + content_tokens + round_input if assistant_token_id is not None else content_tokens + round_input
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        if assistant_token_id is not None:
            total_input.append(assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.model.device)
        return total_input

    @torch.no_grad()
    def chat(self, messages: List[dict], generation_config: Optional[GenerationConfig]=None,**kwargs):
        generation_config = generation_config or self.generation_config
        input_ids = self.build_tokens(messages, generation_config.max_new_tokens)
        self.__class__.generate = PreTrainedModel.generate  # disable stream
        outputs = self.model.generate(input_ids, generation_config=generation_config,**kwargs)
        response = self.post_process(outputs,prompt_length=len(input_ids[0]))
        return response

    @torch.no_grad()
    def chat_stream(self, messages: List[dict], generation_config: Optional[GenerationConfig] = None, **kwargs):
        generation_config = generation_config or self.generation_config
        input_ids = self.build_tokens(messages, generation_config.max_new_tokens)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.generate(input_ids, generation_config=stream_config):
                outputs.append(token.item())
                yield self.tokenizer.decode(outputs, skip_special_tokens=True)

        return stream_generator()
