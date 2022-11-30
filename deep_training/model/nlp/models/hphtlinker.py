# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.norm import LayerNorm

__all__ = [
    'TransformerForHphtlinker'
]

from ..losses.bce_loss import BCELoss


def extract_spoes(outputs):
    for (subject_preds, object_preds) in outputs:
        subject_preds[:, [0, -1]] *= 0
        start = np.where(subject_preds[0, :, 0] > 0.6)[0]
        end = np.where(subject_preds[0, :, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))

class TransformerForHphtlinker(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForHphtlinker, self).__init__(*args,**kwargs)

        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subject_layer = nn.Linear(config.hidden_size, 2)
        self.object_layer = nn.Linear(config.hidden_size, 2 * config.num_labels)

        self.sigmoid = nn.Sigmoid()
        self.condLayerNorm = LayerNorm(hidden_size=config.hidden_size,
                                       conditional_size=config.hidden_size*2)
        self.loss_fn = BCELoss(reduction='sum')

    def get_model_lr(self):
        return super(TransformerForHphtlinker, self).get_model_lr() + [
            (self.subject_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.object_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.condLayerNorm, self.config.task_specific_params['learning_rate_for_task'])
        ]



    def __extract_subject(self,inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs

        start = torch.gather(output, dim=1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, dim=1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]

    def forward_for_stage1(self, **batch):
        # outputs = self(**batch)
        # last_hidden = outputs[0]
        # hidden_output = self.dropout(last_hidden)
        hidden_output = batch['hidden_output']
        subject_preds = self.sigmoid(self.subject_layer(hidden_output)) ** 2
        return subject_preds

    def forward_for_stage2(self, **batch):
        hidden_output = batch['hidden_output']
        subject_ids = batch['subject_ids']
        subject_output = self.__extract_subject([hidden_output, subject_ids])
        subject_output = self.condLayerNorm([hidden_output, subject_output])
        object_preds = self.sigmoid(self.object_layer(subject_output)) ** 4
        object_preds = torch.reshape(object_preds, shape=(*object_preds.shape[:2], self.config.num_labels, 2))
        return object_preds

    def compute_loss(self,batch: dict) -> tuple:
        subject_labels,subject_ids,object_labels = None,None,None
        if 'subject_labels' in batch:
            subject_labels: torch.Tensor = batch.pop('subject_labels')
            subject_ids: torch.Tensor = batch.pop('subject_ids')
            object_labels: torch.Tensor = batch.pop('object_labels')

        outputs = self(**batch)
        hidden_output = outputs[0]
        if self.model.training:
            hidden_output = self.dropout(hidden_output)

        inputs = {k:v for k,v in batch.items()}
        inputs['hidden_output'] = hidden_output
        subject_preds = self.forward_for_stage1(**inputs)


        if subject_labels is not None:
            inputs['subject_ids'] = subject_ids
            object_preds = self.forward_for_stage2(**inputs)
            loss = self.loss_fn(subject_preds, subject_labels) + self.loss_fn(object_preds, object_labels)
            outputs = (loss, subject_preds, object_preds, subject_labels, subject_ids, object_labels)

        else:
            inputs.pop('hidden_output')
            subject_preds[:, [0, -1]] *= 0
            start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
            end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
            subject_ids: list = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subject_ids.append((i, j))

            object_preds = None
            if subject_ids:
                inputs['input_ids'] = np.repeat(inputs['input_ids'], len(subject_ids), 0)
                inputs['token_type_ids'] = np.repeat(inputs['token_type_ids'], len(subject_ids), 0)
                inputs['attention_mask'] = np.repeat(inputs['attention_mask'], len(subject_ids), 0)

                subjects_ids = np.array(subject_ids)
                outputs = self(**batch)
                hidden_output = outputs[0]

                inputs['subjects_ids'] = subjects_ids
                inputs['hidden_output'] = hidden_output
                object_preds = self.forward_for_stage1(**inputs)

            outputs = (subject_preds, object_preds)

        return outputs