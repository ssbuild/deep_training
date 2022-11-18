# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from torch import nn
from .transformer import TransformerModel

__all__ = [
    'TransformerForSplinker'
]



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
            (self.classifier, self.config.task_specific_params['learning_rate']),
        ]

    def compute_loss(self,batch) -> tuple:
        labels = None
        if 'labels' in batch:
            labels: torch.Tensor = batch.pop('labels')
        mask = batch.pop('mask')
        outputs = self(**batch)
        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.sigmoid(self.classifier(logits))
        if labels is not None:
            loss = self.BCELoss(logits=logits, labels=labels, mask=mask)
            logits = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
            outputs = (loss,logits)
        else:
            logits = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
            outputs = (logits,)
        return outputs
