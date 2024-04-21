# -*- coding: utf-8 -*-
# @Time:  19:00
# @Author: tk
# @File：llm_model

import copy
import os
import re
import warnings
from typing import List, Tuple, Optional, Callable
import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration,ChatGLMConfig,setup_model_profile
from deep_training.nlp.models.transformer import TransformerBase
from torch import nn
from transformers import LogitsProcessorList, LogitsProcessor, GenerationConfig, StoppingCriteriaList
from .tokenization_chatglm import ChatGLMTokenizer
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
import logging
logger = logging.getLogger(__name__)




def build_masks_and_position_ids_glm(batch_input_ids, ctxlens):
    max_len = batch_input_ids.size(1)
    batch_position_ids, batch_attention_mask = [], []
    for input_ids,ctxlen in zip(batch_input_ids,ctxlens):
        position_ids = list(range(1,max_len+1))
        assert ctxlen <= max_len
        attention_mask = [1] * ctxlen + [0] * (max_len - ctxlen)
        batch_position_ids.append(torch.tensor(position_ids,dtype=torch.long))
        batch_attention_mask.append(torch.tensor(attention_mask,dtype=torch.bool))

    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_position_ids = torch.stack(batch_position_ids, dim=0)
    return batch_attention_mask,batch_position_ids

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class MyChatGLMForConditionalGeneration(ChatGLMForConditionalGeneration):
    def __init__(self,config):
        super(MyChatGLMForConditionalGeneration, self).__init__(config)


    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs


    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None,logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"logits_processor": logits_processor, **kwargs}

        output_scores = gen_kwargs.get('output_scores', False)
        if output_scores:
            gen_kwargs['return_dict_in_generate'] = True

        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        if output_scores:
            score = outputs.scores[0]
            return score

        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history



class MyTransformerChatGlmLMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(MyTransformerChatGlmLMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyChatGLMForConditionalGeneration, *args, **kwargs))

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
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        self.model.enable_input_require_grads()









class MyTransformer(MyTransformerChatGlmLMHeadModel,ModelWeightMixin, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None,rope_args=None, **kwargs):
        lora_args: PetlArguments = kwargs.pop('lora_args',None)
        num_layers_freeze = kwargs.pop('num_layers_freeze',-1)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.num_layers_freeze = num_layers_freeze
        #可能添加新词
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))

        self.rope_args = rope_args
        inject_rope_scale_layer(self.backbone, rope_args)
        self.inject_model()


    def inject_model(self):
        lora_args = self.lora_args
        num_layers_freeze = self.num_layers_freeze

        # ptv2
        if (self.config.pre_seq_len or 0) > 0:
            self.backbone.enable_input_require_grads()

        if lora_args is not None and lora_args.enable:
            self.backbone.enable_input_require_grads()
            model: PetlModel  = PetlModel(self.backbone, lora_args,
                              auto_prepare_kbit_training=getattr(self,"auto_prepare_kbit_training",True), 
                              use_gradient_checkpointing=getattr(self,"use_gradient_checkpointing", False)
                              )
            print('==' * 30,'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

            # for name, module in model.named_modules():
            #     if isinstance(module, LoraLayer):
            #         module = module.to(torch.bfloat16)
            #     if 'norm' in name:
            #         module = module.to(torch.float32)
            #     if 'lm_head' in name or 'embed_tokens' in name:
            #         if hasattr(module, 'weight'):
            #             if module.weight.dtype == torch.float32:
            #                 module = module.to(torch.bfloat16)

        elif num_layers_freeze > 0 and self.config.pre_seq_len is None:  # 非 lora freeze 非 ptuning模式
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'),param[0])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < num_layers_freeze:
                        param[1].requires_grad = False
                        print('freeze layer',param[0])

    def resize_token_embs(self, new_num_tokens,pad_to_multiple_of=128):
        if new_num_tokens is not None:
            logger.info(f"new_num_tokens:{new_num_tokens}")
            model: PreTrainedModel = self.backbone.model
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if new_num_tokens > embedding_size:
                # lora ptv2 二次加载权重需备份原此词表
                if (self.lora_args is not None and self.lora_args.enable) or (
                        self.prompt_args is not None and self.prompt_args.enable):
                    config = model.config
                    if config.task_specific_params is None:
                        config.task_specific_params = {}
                    config.task_specific_params['vocab_size'] = config.vocab_size

                logger.info("resize the embedding size by the size of the tokenizer")
                # print('before',self.config)
                model.resize_token_embeddings(new_num_tokens,pad_to_multiple_of=pad_to_multiple_of)
                # print('after',self.config)

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.enable:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> MyChatGLMForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        return self.backbone.model