# -*- coding: utf-8 -*-
# @Time    : 2023/1/11 16:52
import typing
from dataclasses import dataclass, field

import torch
from torch import nn
from .transformer import TransformerModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

__all__ = [
    'TransformerForTSDAE'
]

@dataclass
class TsdaelArguments:
    hidden_size: typing.Optional[int] = field(
        default=512,
        metadata={
            "help": (
                ""
            )
        },
    )

    dmodel_type: typing.Optional[str] = field(
        default='bert',
        metadata={
            "help": (
                ""
            )
        },
    )
    dmodel_name_or_path: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )
    dtokenizer_name: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )
    dconfig_name: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )



class TransformerForTSDAE(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForTSDAE, self).__init__(*args,**kwargs)
        tsdae_args:TsdaelArguments = kwargs.pop('tsdae_args')
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, tsdae_args.hidden_size)

        decoder_name_or_path = tsdae_args.dmodel_name_or_path
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(tsdae_args.dtokenizer_name if tsdae_args.dtokenizer_name is not None else decoder_name_or_path)
        if tsdae_args.dconfig_name is not None:
            dconfig_name = tsdae_args.dconfig_name
        elif tsdae_args.dmodel_name_or_path is not None:
            dconfig_name = decoder_name_or_path
        else:
            dconfig_name = tsdae_args.dmodel_type

        self.decoder_config = AutoConfig.from_pretrained(dconfig_name)
        self.need_retokenization = not (type(self.tokenizer_encoder) == type(self.tokenizer_decoder))


        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True

        if decoder_name_or_path is not None:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path,
                                                            from_tf=bool(".ckpt" in decoder_name_or_path),
                                                            config=self.decoder_config)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_config)



    def get_model_lr(self):
        return super(TransformerForTSDAE, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate']),
            (self.decoder, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.classifier(logits)
        tags = self.crf.decode(logits, attention_mask)
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (loss,tags,labels)
        else:
            outputs = (tags,)
        return outputs