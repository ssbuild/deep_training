# @Time    : 2022/11/14 19:53
# @Author  : tk
# @FileName: unilm.py
__all__ = [
    'TransformerForSplinker'
]

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from asmodels.model.nlp.models.transformer import TransformerModel


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

class TransformerForUnilm(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForUnilm, self).__init__(*args, **kwargs)

        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.BCELoss = BCELossForIE()
        self.sigmoid = nn.Sigmoid()


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        attrs = [model, self.classifier]
        opt = []
        for a in attrs:
            opt += [
                {
                    "params": [p for n, p in a.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay, "lr": self.hparams.learning_rate,
                },
                {
                    "params": [p for n, p in a.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": self.hparams.learning_rate,
                },
            ]

        optimizer = AdamW(opt, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

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
        return {"loss": val_loss, "logits": logits.item(), "labels": real_label}

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