# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 12:26
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.crf import CRF
__all__ = [
    'TransformerForCRF'
]

class TransformerForCRF(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForCRF, self).__init__(*args,**kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels)

    def get_model_lr(self):
        return super(TransformerForCRF, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate']),
            (self.crf, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.classifier(logits)
        tags = self.crf.decode(logits, attention_mask)
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (loss,tags,labels)
        else:
            outputs = (tags,)
        return outputs

