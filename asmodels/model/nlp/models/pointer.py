# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer,loss_fn,f1_metric

__all__ = [
    'TransformerPointer'
]

class TransformerPointer(TransformerModel):
    def __init__(self,with_efficient=True, *args,**kwargs):
        super(TransformerPointer, self).__init__(*args,**kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels,64)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        attrs = [model, self.pointer_layer]
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
        outputs = self(**batch)
        logits = outputs[0]

        logits = self.pointer_layer(logits, batch['attention_mask'])

        loss = loss_fn(labels, logits)
        f1 = f1_metric(labels, logits)
        self.log_dict({'train_loss': loss, 'f1': f1}, prog_bar=True)
        return loss