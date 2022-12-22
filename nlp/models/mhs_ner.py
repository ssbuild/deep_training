# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 12:26
import copy
import math

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.mhslayer import MhsLayer
from ..losses.loss_mhslinker import MutiheadlinkerLoss, MutiheadlinkerLossEx

__all__ = [
    'TransformerForMhsNer'
]


def extract_lse(outputs,threshold = 1e-8,top_n=1):
    batch_results = []
    for mhs_logits in outputs:
        mhs_logits[:,[0, -1]] *= 0
        mhs_logits[:,:,[0, -1]] *= 0
        spoes = []
        e_set = {}
        for l,logits in enumerate(mhs_logits):
            e_set.clear()
            for s,e in zip(*np.where(logits > threshold)):
                if s > e:
                    continue
                if s not in e_set:
                    e_set[s] = 0
                if e_set[s] >= top_n:
                    continue
                e_set[s] += 1
                spoes.append((l,s - 1,e - 1))
        batch_results.append(list(set(spoes)))
    return batch_results


class TransformerForMhsNer(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForMhsNer, self).__init__(*args, **kwargs)
        config = self.config

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mhslayer = MhsLayer(config.hidden_size,config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = MutiheadlinkerLossEx()

    def get_model_lr(self):
        return super(TransformerForMhsNer, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.mhslayer, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)

        logits_mhs = self.mhslayer(logits,attention_mask)
        if labels is not None:
            loss_mhs = self.loss_fn(logits_mhs,labels)
            outputs = (loss_mhs,logits_mhs,labels)
        else:
            outputs = (logits_mhs,labels)
        return outputs

