# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/20 10:15
import re
from deep_training.nlp.models.petl import PetlModel,get_prompt_model, PromptModel
from torch import nn
from transformers import PreTrainedModel


import logging
logger = logging.getLogger(__name__)

class BaseModelWrapper:
    def inject_model(self):
        lora_args,prompt_args = self.lora_args,self.prompt_args
        num_layers_freeze = getattr(self,"num_layers_freeze",-1)
        pre_seq_len = getattr(self.config,"pre_seq_len",None)
        if lora_args is not None and lora_args.enable:
            # self.backbone.enable_input_require_grads()
            model: PetlModel = PetlModel(self.backbone, lora_args,
                                         auto_prepare_kbit_training=getattr(self,"auto_prepare_kbit_training",True), 
                                         use_gradient_checkpointing=getattr(self,"gradient_checkpointing",False))
            print('==' * 30, 'lora info')
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

        elif prompt_args is not None and prompt_args.enable:
            self.backbone.enable_input_require_grads()
            model: PromptModel = get_prompt_model(self.backbone, prompt_args)
            print('==' * 30, 'prompt info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

        elif num_layers_freeze > 0 and pre_seq_len is None:  # 非 lora freeze 非 ptuning模式
            M: nn.Module = self.backbone
            for param in M.named_parameters():
                result = re.match(re.compile('.*transformer.layers.(\\d+)'), param[ 0 ])
                if result is not None:
                    n_layer = int(result.group(1))
                    if n_layer < num_layers_freeze:
                        param[ 1 ].requires_grad = False
                        print('freeze layer', param[ 0 ])

    def resize_token_embs(self,new_num_tokens,pad_to_multiple_of=128):
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
        elif self.prompt_args and self.prompt_args.enable:
            return [(self.backbone, lr)]
        return super().get_model_lr(model, lr)


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.enable:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.enable:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model