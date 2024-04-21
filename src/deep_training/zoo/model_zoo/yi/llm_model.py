# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
from typing import Optional
import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.transformer_base import TransformerBase
from deep_training.nlp.models.yi.modeling_yi import YiForCausalLM,YiConfig,setup_model_profile
from transformers.generation.streamers import BaseStreamer

from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
from .tokenization_yi import YiTokenizer

import logging
logger = logging.getLogger(__name__)


class MyYiForCausalLM(YiForCausalLM):

    def build_prompt(self,tokenizer, query, history=None, prefix=None):
        if prefix:
            prompt = "<|im_start|>system\n{}<|im_end|>\n"
        else:
            prompt = ''
        if history is not None:
            for q, a in history:
                prompt += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(q, a)
        prompt += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        return prompt


    @torch.no_grad()
    def chat(self,
             tokenizer,
             query: str,
             history=None,
             prefix=None,
             streamer: Optional[BaseStreamer] = None,
             **kwargs):
        if history is None:
            history = []
        prompt = self.build_inputs(tokenizer, query, history,prefix=prefix)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = self.generate(**inputs,
                                streamer=streamer,
                                **kwargs)
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self,
                    tokenizer,
                    query: str,
                    history=None,
                    **kwargs):
        if history is None:
            history = []

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]
                token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
                print(token, end="")

            def end(self):
                print("")

        return self.chat(
            tokenizer=tokenizer,
            query=query,
            streamer=ChatStreamer(tokenizer=tokenizer),
            history=history,
            **kwargs
        )
class TransformerForLM(TransformerBase):
    def __init__(self, *args, **kwargs):
        super(TransformerForLM, self).__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(MyYiForCausalLM, *args, **kwargs))
        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        # class CastOutputToFloat(nn.Sequential):
        #     def forward(self, x):
        #         return super().forward(x).to(torch.float32)
        #
        # self.model.lm_head = CastOutputToFloat(self.model.lm_head)



    def enable_input_require_grads(self):
        #setattr(self.model, 'model_parallel', True)
        #setattr(self.model, 'is_parallelizable', True)
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


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model

