# coding=utf8
# @Time    : 2023/07/18 10:41
# @Author  : tk
# @FileName: llm_model
import queue
import threading
from typing import List, Tuple, Optional
import torch
from deep_training.nlp.layers.rope_scale.patch import *
from deep_training.nlp.models.internlm2.modeling_internlm import InternLM2ForCausalLM,InternLM2Config,setup_model_profile
from deep_training.nlp.models.transformer_base import TransformerBase
from transformers.generation.streamers import BaseStreamer

from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
from .tokenization_internlm2 import InternLM2Tokenizer
from .tokenization_internlm2_fast import InternLM2Tokenizer as InternLM2Tokenizer_Fast
import logging
logger = logging.getLogger(__name__)


class MyInternLMForCausalLM(InternLM2ForCausalLM):
    def build_inputs(self, tokenizer, query: str, history=None, meta_instruction=""):
        if history is None:
            history = [ ]
        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[ 0 ]}<|im_end|>\n<|im_start|>assistant\n{record[ 1 ]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return tokenizer([ prompt ], return_tensors="pt")

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            query: str,
            history=None,
            streamer: Optional[ BaseStreamer ] = None,
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_p: float = 0.8,
            meta_instruction: str = "You are an AI assistant whose name is InternLM (书生·浦语).\n"
                                    "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
                                    "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
            **kwargs,
    ):
        if history is None:
            history = [ ]
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [ tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids([ "<|im_end|>" ])[ 0 ] ]
        eos_token_id_ = kwargs.pop("eos_token_id",[])
        if not isinstance(eos_token_id_,list):
            eos_token_id_ = [eos_token_id_]
        eos_token_id = list(set(eos_token_id + eos_token_id_))
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[ 0 ].cpu().tolist()[ len(inputs[ "input_ids" ][ 0 ]): ]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<|im_end|>")[ 0 ]
        history = history + [ (query, response) ]
        return response, history

    @torch.no_grad()
    def stream_chat(
            self,
            tokenizer,
            query: str,
            history: List[ Tuple[ str, str ] ] = [ ],
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_p: float = 0.8,
            **kwargs,
    ):
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                "The version of `transformers` is too low. Please make sure "
                "that you have installed `transformers>=4.28.0`."
            )

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.cache = [ ]
                self.received_inputs = False
                self.queue.put((self.response, history + [ (self.query, self.response) ]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[ 0 ] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[ 0 ]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                if token.strip() != "<|im_end|>":
                    self.response = self.response + token
                    history = self.history + [ (self.query, self.response) ]
                    self.queue.put((self.response, history))
                    self.cache = [ ]
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


class TransformerForLM(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerForLM, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(MyInternLMForCausalLM, *args, **kwargs))

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


    def get_llm_model(self) -> MyInternLMForCausalLM:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model





