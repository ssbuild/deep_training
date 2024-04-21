# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/25 14:56
import re
from deep_training.nlp.layers.rope_scale.patch import *
from typing import List, Tuple, Optional,Any,Union
import torch
from torch import nn
from deep_training.nlp.models.xverse.modeling_xverse import XverseForCausalLM,XverseConfig # noqa
from deep_training.nlp.models.transformer import TransformerBase
from transformers import GenerationConfig

from ..auto.base_wapper import BaseModelWrapper
from ...weight.modelweighter import *
from ...utils.transformer_utils import hf_decorator
import logging
logger = logging.getLogger(__name__)


class MyXverseForCausalLM(XverseForCausalLM):
    def _build_chat_input(self, tokenizer, messages: List[dict], max_new_tokens: int = 2048):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = self.config.max_position_embeddings - max_new_tokens
        max_input_tokens = max(self.config.max_position_embeddings // 2, max_input_tokens)

        total_input, round_input = [], []
        user_prompt, assist_prompt = "Human: ", "Assistant: "
        for i, message in enumerate(messages[::-1]):
            if message['role'] == 'user':
                user_content = f"{user_prompt}{message['content']}\n\n"
                if i == 0:
                    user_content += assist_prompt
                content_tokens = tokenizer.encode(user_content, return_token_type_ids=False)
                round_input = content_tokens + round_input

                if i != 0:
                    if len(total_input) + len(round_input) > max_input_tokens:
                        break
                    else:
                        total_input = round_input + total_input
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                round_input = []
            elif message['role'] == 'assistant':
                assist_content = f"{assist_prompt}{message['content']}"
                content_tokens = tokenizer.encode(assist_content, return_token_type_ids=False)
                round_input = content_tokens + [self.generation_config.eos_token_id] + round_input
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        total_input = torch.LongTensor([total_input]).to(self.device)
        return total_input

    @torch.no_grad()
    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig] = None,**kwargs):
        generation_config = generation_config or self.generation_config
        input_ids = self._build_chat_input(tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            from transformers import TextIteratorStreamer
            from threading import Thread
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
            self.__class__.generate = PreTrainedModel.generate

            def stream_generator():
                generation_kwargs = dict(inputs=input_ids, generation_config=generation_config, streamer=streamer,**kwargs)
                thread = Thread(target=self.generate, kwargs=generation_kwargs)
                thread.start()
                for next_text in streamer:
                    yield next_text.rstrip(tokenizer.eos_token)

            return stream_generator()
        else:
            self.__class__.generate = PreTrainedModel.generate  # disable stream
            outputs = self.generate(input_ids, generation_config=generation_config,**kwargs)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response

class TransformerForLM(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyXverseForCausalLM, *args, **kwargs))

    def enable_input_require_grads(self):
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

class MyTransformer(TransformerForLM, ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args

        #可能扩充词表
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))

        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()


    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.enable:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.enable:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)


    def get_llm_model(self) -> Optional[Union[MyXverseForCausalLM,Any]]:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model
