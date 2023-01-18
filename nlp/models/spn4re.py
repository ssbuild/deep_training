# -*- coding: utf-8 -*-
# @Time    : 2023/1/9 10:52
#reference: https://github.com/DianboWork/SPN4RE

import typing
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from .transformer import TransformerModel
from ..losses.loss_spn4re import SetCriterion

__all__ = [
    'TransformerForSPN4RE'
]


@dataclass
class Spn4reArguments:
    num_generated_triples: typing.Optional[int] = field(
        default=20,
        metadata={
            "help": (
                ""
            )
        },
    )
    num_decoder_layers: typing.Optional[int] = field(
        default=3,
        metadata={
            "help": (
                ""
            )
        },
    )
    matcher: typing.Optional[str] = field(
        default='avg',
        metadata={
            "help": (
                "one of ['avg', 'min']"
            )
        },
    )

    na_rel_coef: typing.Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                ""
            )
        },
    )

    rel_loss_weight: typing.Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                ""
            )
        },
    )
    head_ent_loss_weight: typing.Optional[float] = field(
        default=2.0,
        metadata={
            "help": (
                ""
            )
        },
    )
    tail_ent_loss_weight: typing.Optional[float] = field(
        default=2.0,
        metadata={
            "help": (
                ""
            )
        },
    )
    fix_bert_embeddings: typing.Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                ""
            )
        },
    )
    max_span_length: typing.Optional[int] = field(
        default=20,
        metadata={
            "help": (
                ""
            )
        },
    )
    n_best_size: typing.Optional[int] = field(
        default=100,
        metadata={
            "help": (
                ""
            )
        },
    )

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_best_spans(start_logits,end_logits,n_best_size,max_span_length):
    ss = _get_best_indexes(start_logits, n_best_size)
    ee = _get_best_indexes(end_logits, n_best_size)
    spans = set()
    if 0 in ss:
        ss.remove(0)
    if 0 in ee:
        ee.remove(0)
    for i in ss:
        for j in ee:
            if i> j:
                continue
            if j -i + 1 > max_span_length:
                continue
            spans.add((i, j))
    return list(spans)


def softmax(x,axis = None):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x,axis=axis,keepdims=True)
    return softmax_x


def extract_spoes(outputs: typing.List,n_best_size,max_span_length):
    batch_result = []
    for class_logits,head_logits,tail_logits,seqlen in zip(outputs[0],outputs[1],outputs[2],outputs[3]):
        n = class_logits.shape[0]
        class_logits = np.argmax(class_logits,axis=-1)
        spoes = set()
        for i in range(n):
            p = class_logits[i]
            if p == 0:
                continue
            start_logits = softmax(head_logits[i][0][:seqlen],axis=-1)
            end_logigts = softmax(head_logits[i][1][:seqlen],axis=-1)
            subs = get_best_spans(start_logits,end_logigts, n_best_size, max_span_length)
            start_logits = softmax(tail_logits[i][0][:seqlen],axis=-1)
            end_logigts = softmax(tail_logits[i][1][:seqlen],axis=-1)
            objs = get_best_spans(start_logits,end_logigts, n_best_size, max_span_length)
            for s in subs:
                for o in objs:
                    spoes.add((s[0]-1,s[1]-1,p-1,o[0]-1,o[1]-1))
        batch_result.append(spoes)
    return batch_result

class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
        self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        class_logits = self.decoder2class(hidden_states)

        head_start_logits = self.head_start_metric_3(torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        head_end_logits = self.head_end_metric_3(torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        tail_start_logits = self.tail_start_metric_3(torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_end_logits = self.tail_end_metric_3(torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()


        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class TransformerForSPN4RE(TransformerModel):
    def __init__(self, *args,**kwargs):
        spn4re_args: Spn4reArguments = kwargs.pop('spn4re_args')
        super(TransformerForSPN4RE, self).__init__(*args,**kwargs)
        self.spn4re_args = spn4re_args
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        if spn4re_args.fix_bert_embeddings:
            for param in self.model.parameters():
                param.requires_grad = False

        self.decoder = SetDecoder(config,
                                  self.spn4re_args.num_generated_triples,
                                  self.spn4re_args.num_decoder_layers,
                                  self.config.num_labels,
                                  return_intermediate=False)

        self.criterion = SetCriterion(self.config.num_labels,
                                      loss_weight=self.get_loss_weight(spn4re_args),
                                      na_coef=self.spn4re_args.na_rel_coef,
                                      losses=["entity", "relation"],
                                      matcher=self.spn4re_args.matcher)


    def get_model_lr(self):
        return super(TransformerForSPN4RE, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate']),
            (self.decoder, self.config.task_specific_params['learning_rate']),
            (self.criterion, self.config.task_specific_params['learning_rate']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        last_hidden_state = logits

        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(
            encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                      -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                      -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                  -10000.0)  # [bsz, num_generated_triples, seq_len]

        head_logits = torch.stack([head_start_logits,head_end_logits],dim=2)
        tail_logits = torch.stack([tail_start_logits, tail_end_logits], dim=2)
        seqlens = torch.sum(attention_mask,dim=-1)
        if labels is not None:
            outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits,
                       'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits,
                       'tail_end_logits': tail_end_logits}
            loss = self.criterion(outputs, labels)
            outputs = (loss,class_logits,head_logits,tail_logits,seqlens)
        else:
            outputs = (class_logits,head_logits,tail_logits,seqlens)
        return outputs

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight,
                "tail_entity": args.tail_ent_loss_weight}