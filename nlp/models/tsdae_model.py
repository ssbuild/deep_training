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
    vector_size: typing.Optional[int] = field(
        default=512,
        metadata={
            "help": (
                ""
            )
        },
    )

    num_encoder_layer: typing.Optional[int] = field(
        default=6,
        metadata={
            "help": (
                ""
            )
        },
    )

    num_decoder_layer: typing.Optional[int] = field(
        default=6,
        metadata={
            "help": (
                ""
            )
        },
    )

    decoder_model_type: typing.Optional[str] = field(
        default='bert',
        metadata={
            "help": (
                ""
            )
        },
    )
    decoder_model_name_or_path: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )
    decoder_tokenizer_name: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )
    decoder_config_name: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )



class TransformerForTSDAE(TransformerModel):
    def __init__(self, *args,**kwargs):
        tsdae_args: TsdaelArguments = kwargs.pop('tsdae_args')
        super(TransformerForTSDAE, self).__init__(*args,**kwargs)
        self.tsdae_args = tsdae_args
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, tsdae_args.vector_size)
        self.classifier2 = nn.Linear(tsdae_args.vector_size, config.hidden_size)

        decoder_model_name_or_path = tsdae_args.decoder_model_name_or_path
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(tsdae_args.decoder_tokenizer_name if tsdae_args.decoder_tokenizer_name is not None else decoder_model_name_or_path)
        if tsdae_args.decoder_config_name is not None:
            decoder_config_name = tsdae_args.decoder_config_name
        elif tsdae_args.decoder_model_name_or_path is not None:
            decoder_config_name = decoder_model_name_or_path
        else:
            decoder_config_name = tsdae_args.decoder_model_type

        self.decoder_config = AutoConfig.from_pretrained(decoder_config_name)

        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True

        if decoder_model_name_or_path is not None:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name_or_path,
                                                            from_tf=bool(".ckpt" in decoder_model_name_or_path),
                                                            config=self.decoder_config)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.decoder_tokenizer.pad_token_id)


    def get_model_lr(self):
        return super(TransformerForTSDAE, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.classifier2, self.config.task_specific_params['learning_rate_for_task']),
            (self.decoder, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        if labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.startswith('target_'):
                    inputs[k.replace('target_','')] = batch.pop(k)

        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch,output_hidden_states=True)
        logits = outputs[2][-self.tsdae_args.num_encoder_layer][:,0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.classifier(logits)
        if labels is not None:
            # Decode
            decoder_outputs = self.decoder(
                **inputs,
                encoder_hidden_states=logits[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
                encoder_attention_mask=attention_mask[:, 0:1],
                output_hidden_states = True,
                labels=None,
                use_cache=False
            )
            # Calculate loss
            lm_logits = decoder_outputs[2][-self.tsdae_args.num_decoder_layer]
            loss = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.reshape(-1))
            outputs = (loss,logits)
        else:
            outputs = (logits,)
        return outputs