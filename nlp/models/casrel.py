# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.norm import LayerNorm
from ..losses.loss_casrel import LossForCasRel

__all__ = [
    'TransformerForHphtlinker'
]

def extract_spoes(outputs):
    batch_result = []
    for (subject_preds, object_preds) in zip(outputs[0],outputs[1]):
        spoes = []
        subject_preds[[0, -1]] *= 0
        start = np.where(subject_preds[:, 0] > 0.6)[0]
        end = np.where(subject_preds[:, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            for subject, object_pred in zip(subjects, object_preds):
                if len(object_pred) == 0:
                    continue
                object_pred[[0, -1]] *= 0
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append(
                                ((subject[0] - 1,subject[1] -1, predicate1, _start - 1,_end -1 ))
                            )
                            break
        batch_result.append(spoes)
    return batch_result

class TransformerForHphtlinker(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForHphtlinker, self).__init__(*args,**kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subject_layer = nn.Linear(config.hidden_size, 2)
        self.object_layer = nn.Linear(config.hidden_size, 2 * config.num_labels)

        self.sigmoid = nn.Sigmoid()
        self.cond_norm_layer = LayerNorm(hidden_size=config.hidden_size,
                                         cond_dim=config.hidden_size * 2)
        self.loss_fn = LossForCasRel(reduction='none')

    def get_model_lr(self):
        return super(TransformerForHphtlinker, self).get_model_lr() + [
            (self.subject_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.object_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.cond_norm_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.sigmoid, self.config.task_specific_params['learning_rate_for_task']),
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def _extract_subject_hidden(self, inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        h =  output.shape[-1]
        start = torch.gather(output, 1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, h))
        end = torch.gather(output, 1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, h))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]

    def forward_for_net(self, *args,**batch):
        outputs = self.model(*args,**batch)
        hidden_output = outputs[0]
        if self.training:
            hidden_output = self.dropout(hidden_output)
        return hidden_output

    def forward_for_subject(self, **batch):
        hidden_output = batch['hidden_output']
        subject_preds = self.sigmoid(self.subject_layer(hidden_output)) ** 2
        return subject_preds

    def forward_for_object(self, hidden_output, subject_ids):
        subject_output = self._extract_subject_hidden([hidden_output, subject_ids])
        subject_output = self.cond_norm_layer([hidden_output, subject_output])
        object_preds = self.sigmoid(self.object_layer(subject_output)) ** 4
        object_preds = torch.reshape(object_preds, shape=object_preds.shape[:2] + (self.config.num_labels, 2))
        return object_preds


    def predict_objects(self,hidden_outputs,subject_preds):
        object_preds_list = []
        subject_preds[:, [0, -1]] *= 0
        starts = subject_preds[:, :, 0] > 0.6
        ends = subject_preds[:, :, 1] > 0.5

        starts = starts.cpu().numpy()
        ends = ends.cpu().numpy()
        for start,end,hidden_output in zip(starts,ends,hidden_outputs):
            start = np.where(start)[0]
            end = np.where(end)[0]
            subject_ids = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subject_ids.append((i, j))
            if subject_ids:
                subject_ids = torch.tensor(subject_ids,dtype=torch.long).to(hidden_output.device)
                hidden_output = torch.repeat_interleave(hidden_output.unsqueeze(0),len(subject_ids),0)
                object_preds = self.forward_for_object(hidden_output, subject_ids)
                object_preds_list.append(object_preds)
            else:
                object_preds_list.append(torch.zeros(size=(*hidden_output.shape[:2],self.config.num_labels,2)))
        return object_preds_list

    def compute_loss(self,*args,**batch) -> tuple:
        subject_labels,subject_ids,object_labels = None,None,None
        if 'subject_labels' in batch:
            subject_labels: torch.Tensor = batch.pop('subject_labels')
            subject_ids: torch.Tensor = batch.pop('subject_ids')
            object_labels: torch.Tensor = batch.pop('object_labels')

        hidden_output = self.forward_for_net(*args,**batch)
        inputs = {k:v for k,v in batch.items()}
        inputs['hidden_output'] = hidden_output
        subject_preds = self.forward_for_subject(**inputs)
        if subject_labels is not None:
            if self.training:
                attention_mask = inputs['attention_mask']
                object_preds = self.forward_for_object(hidden_output, subject_ids)
                loss = self.loss_fn((subject_preds, object_preds),(subject_labels, object_labels),attention_mask)
                outputs = (loss, subject_preds, object_preds, subject_labels, object_labels)
            else:
                object_preds_list = self.predict_objects(hidden_output,subject_preds)
                outputs = (None, subject_preds, object_preds_list, subject_labels, object_labels)
        else:
            object_preds_list = self.predict_objects(hidden_output,subject_preds)
            outputs = (subject_preds, object_preds_list)
        return outputs