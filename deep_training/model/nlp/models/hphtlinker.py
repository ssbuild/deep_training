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

class BCELossForHphtLinker(nn.Module):
    def __init__(self, reduction='none'):
        super(BCELossForHphtLinker, self).__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, subject_preds, subject_labels,object_preds, object_labels,mask=None):
        loss1 = self.criterion(subject_preds, subject_labels)
        loss2 = self.criterion(object_preds, object_labels)
        if mask is None:
            mask = torch.ones((),dtype=torch.float32).to(subject_preds.device)


        loss1 = loss1.mean(2)
        loss1 = loss1 * mask / torch.sum(mask)
        loss1 = loss1.mean()

        loss2 = loss2.sum(3).sum(2)
        loss2 = loss2 * mask / torch.sum(mask)
        loss2 = loss2.mean()
        return loss1+ loss2 ,loss1,loss2


def extract_spoes(outputs):
    batch_result = []
    for (subject_preds, object_preds) in zip(outputs[0],outputs[1]):
        spoes = []
        print('subject.......',type(subject_preds),subject_preds.shape)
        print('object.......',type(object_preds),object_preds.shape)
        subject_preds[[0, -1]] *= 0
        start = np.where(subject_preds[:, 0] > 0.6)[0]
        end = np.where(subject_preds[:, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        print(subjects)
        if subjects:
            object_preds[[0, -1]] *= 0
            for subject, object_pred in zip(subjects, object_preds):
                if object_pred is None:
                    continue
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append(
                                ((subject[0],subject[1], predicate1, _start,_end))
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
        self.condLayerNorm = LayerNorm(hidden_size=config.hidden_size,
                                       conditional_size=config.hidden_size*2)
        self.loss_fn = BCELossForHphtLinker()

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

    def forward_for_net(self,  **batch):
        outputs = self(**batch)
        hidden_output = outputs[0]
        if self.model.training:
            hidden_output = self.dropout(hidden_output)
        return hidden_output

    def forward_for_stage1(self, **batch):

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

        hidden_output = self.forward_for_net(**batch)
        inputs = {k:v for k,v in batch.items()}
        inputs['hidden_output'] = hidden_output
        subject_preds = self.forward_for_stage1(**inputs)
        if subject_labels is not None:
            attention_mask = inputs['attention_mask']

            inputs['subject_ids'] = subject_ids
            object_preds = self.forward_for_stage2(**inputs)
            loss,loss1,loss2 = self.loss_fn(subject_preds, subject_labels,object_preds, object_labels)
            loss_dict = {
                "loss_subject": loss1,
                "loss_object_p": loss2,
                "loss":  loss,
            }
            outputs = (loss_dict, subject_preds, object_preds, subject_labels, subject_ids, object_labels)

        else:
            b_input_ids = inputs['input_ids']
            b_attention_mask = inputs['attention_mask']
            object_preds_list = []
            subject_preds[:, [0, -1]] *= 0
            starts = subject_preds[:, :, 0] > 0.6
            ends = subject_preds[:, :, 1] > 0.5
            for input_ids,attention_mask in zip(b_input_ids,b_attention_mask):
                for i in range(len(starts)):
                    start = np.where(starts[i])[0]
                    end = np.where(ends[i])[0]

                    subjects_ids = []
                    for i in start:
                        j = end[end >= i]
                        if len(j) > 0:
                            j = j[0]
                            subjects_ids.append((i, j))


                    if subjects_ids:
                        input_ids = np.repeat(input_ids, len(subjects_ids), 0)
                        attention_mask = np.repeat(attention_mask, len(subjects_ids), 0)
                        subjects_ids = np.array(subjects_ids)

                        inputs2 = {}
                        inputs2['input_ids'] = input_ids
                        inputs2['attention_mask'] = attention_mask
                        hidden_output = self.forward_for_net(**inputs)
                        inputs2['hidden_output'] = hidden_output
                        inputs2['subjects_ids'] = subjects_ids
                        #seq,num_labels,2
                        object_preds = self.forward_for_stage2(**inputs)
                        object_preds_list.append(object_preds[0])
                    else:
                        object_preds_list.append(None)
            outputs = (subject_preds, object_preds_list)

        return outputs