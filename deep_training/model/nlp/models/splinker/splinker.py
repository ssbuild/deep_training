# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import numpy as np
import torch
from torch import nn
from ..transformer import TransformerModel

__all__ = [
    'TransformerForSplinker'
]



def get_spoes(logits_all,seq_len_all,id2labels):
    batch_result = []
    for (i, (logits, seq_len)) in enumerate(zip( logits_all, seq_len_all)):
        logits = logits[1:seq_len + 1]  # slice between [CLS] and [SEP] to get valid logits
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        # 12
        num_real_label = (len(id2labels) - 2) // 2
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label < num_real_label and (cls_label + num_real_label - 2) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))
        batch_result.append(subject_id_list)
    return batch_result


class BCELossForIE(nn.Module):
    def __init__(self, ):
        super(BCELossForIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.mean(loss,dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

class TransformerForSplinker(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForSplinker, self).__init__(*args, **kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.BCELoss = BCELossForIE()
        self.sigmoid = nn.Sigmoid()

    def get_model_lr(self):
        return super(TransformerForSplinker, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self,batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        mask = batch.pop('mask')
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.sigmoid(self.classifier(logits))

        seqlen = torch.sum(attention_mask,dim=1,keepdim=False).long() -2
        if labels is not None:
            loss = self.BCELoss(logits=logits, labels=labels, mask=mask)
            tags = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
            outputs = (loss,tags,seqlen,labels)
        else:
            tags = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
            outputs = (tags,seqlen,)
        return outputs
