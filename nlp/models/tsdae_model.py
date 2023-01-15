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
    pooling: typing.Optional[str] = field(
        default='reduce',
        metadata={
            "help": (
                "one of [cls , reduce]"
            )
        },
    )

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
        decoder_config = kwargs.pop('decoder_config',None)
        decoder_tokenizer = kwargs.pop('decoder_tokenizer',None)
        super(TransformerForTSDAE, self).__init__(*args,**kwargs)
        self.tsdae_args = tsdae_args
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, tsdae_args.vector_size)


        self.decoder_config = decoder_config
        self.decoder_tokenizer = decoder_tokenizer
        if decoder_config is not None:
            self.classifier2 = nn.Linear(tsdae_args.vector_size, config.hidden_size)

            self.decoder_config.is_decoder = True
            self.decoder_config.add_cross_attention = True

            model_name_or_path = tsdae_args.decoder_model_name_or_path
            if model_name_or_path is not None:
                self.decoder = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                    from_tf=bool(".ckpt" in model_name_or_path),
                                                                    config=self.decoder_config)
            else:
                self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_config)

            self.predictions = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.decoder_tokenizer.pad_token_id)
        else:
            self.predictions, self.classifier2,self.decoder,self.loss_fn = None,None,None,None



    def get_model_lr(self):
        return super(TransformerForTSDAE, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.classifier2, self.config.task_specific_params['learning_rate_for_task']),
            (self.decoder, self.config.task_specific_params['learning_rate_for_task']),
            (self.predictions, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def forward_for_encoder(self,*args, **inputs):
        outputs = self.model(*args, **inputs, output_hidden_states=True)

        # simcse_logits = outputs[0][:, 0]
        # if self.pooling == 'cls':
        #     simcse_logits = outputs[0][:, 0]
        # elif self.pooling == 'pooler':
        #     simcse_logits = outputs[1]
        # elif self.pooling == 'last-avg':
        #     last = outputs[0].transpose(1, 2)  # [batch, 768, seqlen]
        #     return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        # elif self.pooling == 'first-last-avg':
        #     first = outputs[2][1].transpose(1, 2)  # [batch, 768, seqlen]
        #     last = outputs[2][-1].transpose(1, 2)  # [batch, 768, seqlen]
        #     first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        #     last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        #     avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
        #     return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        # elif self.pooling == 'reduce':
        #     simcse_logits = self.sim_head(outputs[1])
        #     simcse_logits = torch.tanh(simcse_logits)
        # return simcse_logits

        #cls
        logits = outputs[2][self.tsdae_args.num_encoder_layer][:, 0]
        if self.tsdae_args.pooling == 'reduce':
            logits = self.classifier(logits)
            logits = torch.tanh(logits)
        return logits

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        if self.training or labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2','')] = batch.pop(k)

        attention_mask = batch['attention_mask']
        logits = self.forward_for_encoder(*args,**batch)
        if self.training:
            if self.tsdae_args.pooling == 'reduce':
                logits2 = self.classifier2(logits)
                logits2 = torch.tanh(logits2)
            else:
                logits2 = logits

            decoder_input_ids = inputs['input_ids']
            decoder_attention_mask = inputs['attention_mask']

            seqlens = torch.sum(decoder_attention_mask,dim=-1)
            labels = torch.clone(decoder_input_ids)[:,1:]

            for seqlen,ids,mask in zip(seqlens,decoder_input_ids,decoder_attention_mask):
                ids[seqlen -1:] = self.decoder_tokenizer.pad_token_id
                ids[seqlen -2] = self.decoder_tokenizer.sep_token_id
                mask[seqlen -2:] = 0

            decoder_input_ids = decoder_input_ids[:,:-1]
            decoder_attention_mask = decoder_attention_mask[:,:-1]
            inputs['input_ids'] = decoder_input_ids
            inputs['attention_mask'] = decoder_attention_mask

            # Decode
            decoder_outputs = self.decoder(
                **inputs,
                encoder_hidden_states=logits2[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
                encoder_attention_mask=attention_mask[:, 0:1],
                output_hidden_states=True,
                labels=None,
                use_cache=False
            )
            # Calculate loss
            output_hidden = decoder_outputs[1][self.tsdae_args.num_decoder_layer]
            lm_logits = self.predictions(output_hidden)
            loss = self.loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.reshape(-1))
            outputs = (loss, logits)
        #评估
        elif labels is not None:
            logits2 = self.forward_for_encoder(*args,**inputs)
            outputs = (None,logits,logits2,labels)
        else:
            outputs = (logits,)
        return outputs