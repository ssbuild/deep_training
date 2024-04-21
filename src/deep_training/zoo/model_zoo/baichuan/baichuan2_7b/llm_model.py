# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
import typing
from typing import Optional, List,Union,Any
import torch
from deep_training.nlp.models.baichuan2_7b.modeling_baichuan import BaichuanForCausalLM,BaichuanConfig,setup_model_profile
from deep_training.nlp.models.transformer_base import TransformerBase
from transformers import GenerationConfig

from ...auto.base_wapper import BaseModelWrapper
from ....utils.transformer_utils import hf_decorator
from ....weight.modelweighter import *
from .tokenization_baichuan import BaichuanTokenizer
from .generation_utils import build_chat_input
import logging
logger = logging.getLogger(__name__)


class MyBaichuanForCausalLM(BaichuanForCausalLM):
    def _build_chat_input(self, tokenizer, messages: List[dict], max_new_tokens: int=0):
        return build_chat_input(self,tokenizer,messages,max_new_tokens)

    @torch.no_grad()
    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None,**kwargs):
        generation_config = generation_config or self.generation_config
        input_ids = self._build_chat_input(tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
            self.__class__.generate = NewGenerationMixin.generate
            self.__class__.sample_stream = NewGenerationMixin.sample_stream
            stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True,**kwargs)

            def stream_generator():
                outputs = []
                for token in self.generate(input_ids, generation_config=stream_config,**kwargs):
                    outputs.append(token.item())
                    yield tokenizer.decode(outputs, skip_special_tokens=True)

            return stream_generator()
        else:
            self.__class__.generate = PreTrainedModel.generate  # disable stream
            outputs = self.generate(input_ids, generation_config=generation_config,**kwargs)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response



class TransformerForLM(TransformerBase):
    def __init__(self, *args, **kwargs):
        super(TransformerForLM, self).__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(MyBaichuanForCausalLM, *args, **kwargs))
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
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        #可能扩充词表
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
        self.inject_model()


