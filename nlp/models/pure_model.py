# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 9:02
#reference: https://github.com/princeton-nlp/PURE

import typing
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from .transformer import TransformerModel
from ..utils.nlputils import batched_index_select
__all__ = [
    'TransformerForPure'
]

@dataclass
class PureModelArguments:
    max_span_length: typing.Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "spans w/ length up to max_span_length are considered as candidates"
            )
        },
    )
    head_hidden_dim: typing.Optional[int] = field(
        default=150,
        metadata={
            "help": (
                ""
            )
        },
    )
    width_embedding_dim: typing.Optional[int] = field(
        default=150,
        metadata={
            "help": (
                ""
            )
        },
    )

def extract_lse(outputs):
    batch_result = []
    for logits,span,span_mask in zip(np.argmax(outputs[0],axis=-1),outputs[1],outputs[2]):
        seq_map = {}
        for seq,(pt,s_mask) in enumerate(zip(span,span_mask)):
            if s_mask > 0:
                seq_map[seq] = tuple(pt[:2])
        lse = []
        for seq in np.nonzero(logits)[0]:
            if seq in seq_map:
                pt = seq_map[seq]
                lse.append((logits[seq] - 1,pt[0]-1,pt[1] - 1))
        batch_result.append(lse)
    return batch_result


class TransformerForPure(TransformerModel):
    def __init__(self, *args,**kwargs):
        puremodel_args: PureModelArguments = kwargs.pop('puremodel_args')
        super(TransformerForPure, self).__init__(*args,**kwargs)

        self.puremodel_args = puremodel_args

        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(self.puremodel_args.max_span_length + 1, self.puremodel_args.width_embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=config.hidden_size * 2 + self.puremodel_args.width_embedding_dim,
                      out_features = self.puremodel_args.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(self.puremodel_args.head_hidden_dim, config.num_labels + 1)
        )
        self.loss_fn = CrossEntropyLoss(reduction='sum',ignore_index=-100)

    def get_model_lr(self):
        return super(TransformerForPure, self).get_model_lr() + [
            (self.width_embedding, self.config.task_specific_params['learning_rate']),
            (self.classifier, self.config.task_specific_params['learning_rate']),
        ]

    def _get_span_embeddings(self, sequence_output, spans):
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        spans: torch.Tensor = batch.pop('spans')
        spans_mask: torch.Tensor = batch.pop('spans_mask')
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        spans_embedding = self._get_span_embeddings(logits,spans)
        logits = self.classifier(spans_embedding)
        if labels is not None:
            labels = labels.long()
            device = logits.device
            if spans_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                ignore_value = torch.ones_like(labels,device=device,dtype=torch.long) * self.loss_fn.ignore_index
                active_labels = torch.where(active_loss, labels.view(-1), ignore_value.view(-1))
                loss = self.loss_fn(active_logits, active_labels) / logits.size(0)
            else:
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))  / logits.size(0)
            outputs = (loss,logits,spans,spans_mask,labels)
        else:
            outputs = (logits,spans,spans_mask)
        return outputs