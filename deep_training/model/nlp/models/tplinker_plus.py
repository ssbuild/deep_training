# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 9:27

import typing
import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.handshakingkernel import HandshakingKernel

__all__ = [
    'TransformerForTplinker_plus'
]

from ..losses.loss_tplinker import TplinkerLoss


class TransformerForTplinker_plus(TransformerModel):
    def __init__(self,  *args, **kwargs):
        shaking_type = kwargs.get('shaking_type')
        inner_enc_type = kwargs.get('inner_enc_type')
        super(TransformerForTplinker_plus, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.handshakingkernel = HandshakingKernel(self.config.hidden_size,shaking_type,inner_enc_type)

        self.ent_fc = nn.Linear(self.config.hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(self.config.hidden_size, 3) for _ in range(self.config.num_labels)]
        self.tail_rel_fc_list = [nn.Linear(self.config.hidden_size, 3) for _ in range(self.config.num_labels)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
        self.loss_fn = TplinkerLoss()

    def get_model_lr(self):
        return super(TransformerForTplinker_plus, self).get_model_lr() + [
            (self.handshakingkernel, self.config.task_specific_params['learning_rate_for_task']),
            (self.ent_fc, self.config.task_specific_params['learning_rate_for_task']),
            (layer, self.config.task_specific_params['learning_rate_for_task'] for layer in self.head_rel_fc_list),
            (layer, self.config.task_specific_params['learning_rate_for_task'] for layer in self.tail_rel_fc_list),
        ]

    def compute_loss(self, batch):
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits, attention_mask)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        # b s*s/2,3
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        # b s*s/2,3
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        if entity_labels is not None:
            loss1 = self.loss_fn(ent_shaking_outputs, entity_labels)
            loss2 = self.loss_fn(head_rel_shaking_outputs, head_labels)
            loss3 = self.loss_fn(tail_rel_shaking_outputs, tail_labels)
            loss = (loss1 + loss2 + loss3) / 3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_head': loss2,
                         'loss_tail': loss3}
            outputs = (loss_dict, ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs,
                       entity_labels, head_labels, tail_labels)
        else:
            outputs = (ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs)
        return outputs