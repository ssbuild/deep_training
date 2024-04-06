# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/17 14:23
import copy
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable
import requests
import torch
from torch import nn
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.visualglm.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig, \
    setup_model_profile, ChatGLMForConditionalGenerationWithImage  # noqa
from deep_training.nlp.models.transformer import TransformerBase
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig, StoppingCriteriaList
from deep_training.nlp.models.visualglm.visual import BlipImageEvalProcessor
from PIL import Image
from io import BytesIO

from .generation_utils import build_masks_and_position_ids_glm
from .tokenization_chatglm import ChatGLMTokenizer
from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)




class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class MyChatGLMForConditionalGenerationWithImage(ChatGLMForConditionalGenerationWithImage):
    def __init__(self,config):
        super(MyChatGLMForConditionalGenerationWithImage, self).__init__(config)

    @staticmethod
    def process_image(text, image=None):
        '''Process image in text.
        Args:
            text: str, text.
            image: Optional, image path / url / PIL image.
        '''
        image_position = text.rfind("<img>") + 5
        # extract path from <img></img> using re
        image_path = re.findall(r"<img>(.*?)</img>", text)
        image_path = image_path[-1] if image_path else None
        if image_path is not None:
            assert image is None, "image and image_path cannot be both not None."
            text = text.replace(f"<img>{image_path}</img>", "<img></img>")
            # url
            if image_path.startswith("http"):
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content))
            # local path
            else:
                image = Image.open(image_path)
        if image is not None:
            processor = BlipImageEvalProcessor(224)
            image = processor(image.convert('RGB'))
            image = image.unsqueeze(0)
        return text, image_position, image

    def build_inputs_with_image(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None):
        image_path = image_path.strip()
        if image_path:
            prompt = "<img>{}</img>".format(image_path)
        else:
            prompt = ""
        for i, (old_query, response) in enumerate(history):  # history removes image urls/paths, while query does not.
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)
        prompt, image_position, torch_image = self.process_image(prompt)
        if torch_image is not None:
            torch_image = torch_image.to(self.dtype).to(self.device)
            input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
            input1 = [tokenizer.unk_token_id] * self.image_length
            input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
            inputs = sum([input0, input1, input2], [])
            inputs = {
                "input_ids": torch.tensor([tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
                    self.device),
                "pre_image_length": len(input0),
                "images": torch_image}
        else:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(self.device)
            inputs["pre_image_length"] = 0
        return inputs

    @torch.no_grad()
    def chat(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs_with_image(tokenizer, image_path, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None,
                    logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs_with_image(tokenizer, image_path, query, history=history)
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

class MyTransformerChatGlmLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(MyTransformerChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGenerationWithImage, *args, **kwargs))

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
        self.model.enable_input_require_grads()



class MyTransformer(MyTransformerChatGlmLMHeadModel,ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: PetlArguments = kwargs.pop('lora_args',None)
        num_layers_freeze = kwargs.pop('num_layers_freeze',-1)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = None
        self.num_layers_freeze = num_layers_freeze
        #可能添加新词
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
        return super(MyTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyChatGLMForConditionalGenerationWithImage:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        return self.backbone.model
