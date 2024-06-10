# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/17 14:23
import copy
import json
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable, Dict
import requests
import torch
from torch import nn
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.glm4v.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig, \
    setup_model_profile, ChatGLMForConditionalGenerationWithImage  # noqa
from deep_training.nlp.models.transformer import TransformerBase
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig, StoppingCriteriaList
from deep_training.nlp.models.visualglm.visual import BlipImageEvalProcessor
from PIL import Image
from io import BytesIO

from .generation_utils import build_masks_and_position_ids_glm
from .tokenization_chatglm import ChatGLM4Tokenizer
from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)




class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 198] = 5e4
        return scores

class MyChatGLMForConditionalGenerationWithImage(ChatGLMForConditionalGenerationWithImage):
    def __init__(self,config):
        super(MyChatGLMForConditionalGenerationWithImage, self).__init__(config)

    def process_response(self, output, history):
        content = ""
        history = copy.deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[ 0 ][ "role" ] == "system" and "tools" in history[ 0 ]:
                    parameters = json.loads(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[ Dict ] = None, role: str = "user", image=None,
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        if history is None:
            history = [ ]
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        message = {"role": role, "content": query}
        if image is not None:
            message[ "image" ] = image
        history.append(message)
        inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt", return_dict=True)
        inputs = inputs.to(self.device)
        eos_token_id = [ tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|user|>"),
                         tokenizer.convert_tokens_to_ids("<|observation|>") ]
        outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[ 0 ][ len(inputs[ "input_ids" ][ 0 ]):-1 ]
        response = tokenizer.decode(outputs)
        response, history = self.process_response(response, history)
        return response, history

    @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, history: List[ Dict ] = None, role: str = "user", image=None,
                    past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        if history is None:
            history = [ ]
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        eos_token_id = [ tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|user|>"),
                         tokenizer.convert_tokens_to_ids("<|observation|>") ]
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        message = {"role": role, "content": "query"}
        if image is not None:
            message[ "image" ] = image
        if past_key_values is None:
            inputs = tokenizer.apply_chat_template(history + [ message ],
                                                   add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                   return_dict=True)
        else:
            inputs = tokenizer.apply_chat_template([ message ], add_special_tokens=False,
                                                   add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                                   return_dict=True)
        inputs = inputs.to(self.device)
        if past_key_values is not None:
            past_length = past_key_values[ 0 ][ 0 ].shape[ 2 ]
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            inputs.position_ids += past_length
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs[ 'attention_mask' ] = attention_mask
        history.append({"role": role, "content": query})
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                            **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[ 0 ][ len(inputs[ "input_ids" ][ 0 ]):-1 ]
            response = tokenizer.decode(outputs)
            if response and response[ -1 ] != "�":
                response, new_history = self.process_response(response, history)
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @torch.inference_mode()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[ GenerationConfig ] = None,
            logits_processor: Optional[ LogitsProcessorList ] = None,
            stopping_criteria: Optional[ StoppingCriteriaList ] = None,
            prefix_allowed_tokens_fn: Optional[ Callable[ [ int, torch.Tensor ], List[ int ] ] ] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[ 0 ], input_ids.shape[ -1 ]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        model_kwargs[ "use_cache" ] = generation_config.use_cache
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [ eos_token_id ]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[ 0 ]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[ :, -1, : ]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([ input_ids, next_tokens[ :, None ] ], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[ 0 ], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

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
