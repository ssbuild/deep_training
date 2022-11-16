# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel

__all__ = [
    'TransformerForSplinker'
]

from ..utils import configure_optimizers


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

    def configure_optimizers(self):
        attrs = [(self.model,self.config.task_specific_params['learning_rate']),
                 (self.classifier,self.config.task_specific_params['learning_rate_for_task']),]
        return configure_optimizers(attrs, self.hparams,self.trainer.estimated_stepping_batches)


    def training_step(self, batch, batch_idx):
        labels: torch.Tensor = batch.pop('labels')
        mask = batch.pop('mask')
        outputs = self(**batch)
        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.sigmoid(self.classifier(logits))
        loss = self.BCELoss(logits=logits, labels=labels, mask=mask)
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