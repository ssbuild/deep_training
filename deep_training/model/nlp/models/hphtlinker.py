# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel
from ..layers.norm import LayerNorm

__all__ = [
    'TransformerForHphtlinker'
]


class BCELossForLinker(nn.Module):
    def __init__(self,):
        super(BCELossForLinker, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask=None):
        loss = self.criterion(logits, labels)
        if mask is not None:
            mask = mask.float()
            loss = loss * mask.unsqueeze(-1)
            loss = torch.sum(torch.mean(loss,dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

class TransformerForHphtlinker(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForHphtlinker, self).__init__(*args,**kwargs)

        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.subject_layer = nn.Linear(config.hidden_size, 2)
        self.object_layer = nn.Linear(config.hidden_size, 2 * config.num_labels)
        self.BCELoss = BCELossForLinker()
        self.sigmoid = nn.Sigmoid()
        self.condLayerNorm = LayerNorm(hidden_size=config.hidden_size,
                                       conditional_size=config.hidden_size*2)


    def get_model_lr(self):
        return super(TransformerForHphtlinker, self).get_model_lr() + [
            (self.subject_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.object_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]



    def extract_subject(self,inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs

        start = torch.gather(output, dim=1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, dim=1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]


    def training_step(self, batch, batch_idx):
        subject_labels: torch.Tensor = batch.pop('subject_labels')
        subject_ids: torch.Tensor = batch.pop('subject_ids')
        object_labels: torch.Tensor = batch.pop('object_labels')

        outputs = self(**batch)
        last_hidden = outputs[0]
        logits = self.dropout(last_hidden)
        subject_preds = self.sigmoid(self.subject_layer(logits)) ** 2


        subject_output = self.extract_subject([last_hidden,subject_ids])
        subject_output = self.condLayerNorm([last_hidden,subject_output])
        object_preds = self.sigmoid(self.object_layer(subject_output)) ** 4
        object_preds = torch.reshape(object_preds,shape=(*object_preds.shape[:2],self.config.num_labels,2))

        loss = self.BCELoss(subject_preds,subject_labels) + self.BCELoss(object_preds,object_labels)

        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels: torch.Tensor = batch.pop('labels')
        real_label = batch.pop("real_label")
        mask = batch.pop("mask")
        outputs = self(**batch)
        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        logits = self.sigmoid(logits)
        val_loss = self.BCELoss(logits=logits, labels=labels, mask=mask)
        logits = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
        return {"losses": val_loss, "logits": logits.item(), "labels": real_label}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        outputs = self(x)
        logits = outputs[0]

        logits = self.dropout(logits)
        logits = self.classifier(logits)

        logits = self.sigmoid(logits)
        logits = torch.where(logits >= 0.5, torch.ones_like(logits), torch.zeros_like(logits))
        return logits