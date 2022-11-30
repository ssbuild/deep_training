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

    def forward_for_subject(self, **batch):
        subject_labels, subject_ids, object_labels = None, None, None
        if 'subject_labels' in batch:
            subject_labels: torch.Tensor = batch.pop('subject_labels')
            subject_ids: torch.Tensor = batch.pop('subject_ids')
            object_labels: torch.Tensor = batch.pop('object_labels')

        outputs = self(**batch)
        last_hidden = outputs[0]
        logits = self.dropout(last_hidden)
        subject_preds = self.sigmoid(self.subject_layer(logits)) ** 2

    def forward_for_objectp(self, **inputs):...

    def compute_loss(self,batch) -> tuple:
        subject_labels,subject_ids,object_labels = None,None,None
        if 'subject_labels' in batch:
            subject_labels: torch.Tensor = batch.pop('subject_labels')
            subject_ids: torch.Tensor = batch.pop('subject_ids')
            object_labels: torch.Tensor = batch.pop('object_labels')

        outputs = self(**batch)
        last_hidden = outputs[0]
        logits = self.dropout(last_hidden)
        subject_preds = self.sigmoid(self.subject_layer(logits)) ** 2

        if subject_labels is not None:
            subject_output = self.__extract_subject([last_hidden, subject_ids])
            subject_output = self.condLayerNorm([last_hidden, subject_output])

            object_preds = self.sigmoid(self.object_layer(subject_output)) ** 4
            object_preds = torch.reshape(object_preds, shape=(*object_preds.shape[:2], self.config.num_labels, 2))

            loss = self.BCELoss(subject_preds, subject_labels) + self.BCELoss(object_preds, object_labels)

            outputs = (loss,subject_preds,object_preds,subject_labels,subject_ids,object_labels)
        else:
            subject_preds[:, [0, -1]] *= 0
            start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
            end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
            subjects = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subjects.append((i, j))

            subject_ids = torch.tensor(subjects,dtype=torch.int64).to(subject_preds.device)

            subject_output = self.__extract_subject([last_hidden, subject_ids])
            subject_output = self.condLayerNorm([last_hidden, subject_output])

            object_preds = self.sigmoid(self.object_layer(subject_output)) ** 4
            object_preds = torch.reshape(object_preds, shape=(*object_preds.shape[:2], self.config.num_labels, 2))
            outputs = (subject_preds, object_preds)

        return outputs