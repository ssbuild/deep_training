# @Time    : 2023/4/2 23:14
# @Author  : tk
# @FileName: models

from deep_training.nlp.models.transformer import TransformerForSeq2SeqLM

from ..auto.base_wapper import BaseModelWrapper
from ...utils.transformer_utils import hf_decorator
from ...weight.modelweighter import *
from transformers import T5ForConditionalGeneration
import logging
logger = logging.getLogger(__name__)

class TransformerForLM(TransformerForSeq2SeqLM):
    def __init__(self, *args, **kwargs):
        super(TransformerForLM, self).__init__(*args, **kwargs)


    def enable_input_require_grads(self):
        # setattr(self.model, 'model_parallel', True)
        # setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()




class MyTransformer(TransformerForLM,ModelWeightMixin,BaseModelWrapper, with_pl=True):
    @hf_decorator
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        self.resize_token_embs(new_num_tokens,getattr(self,"pad_to_multiple_of",128))
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

    def get_llm_model(self) -> T5ForConditionalGeneration:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            # PromptModel 方法覆盖原来方法
            return self.backbone.model
        return self.backbone.model

