# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer,loss_fn,f1_metric

__all__ = [
    'TransformerForPointer'
]


class TransformerForPointer(TransformerModel):
    def __init__(self,config, train_args,with_efficient=True, *args,**kwargs):
        super(TransformerForPointer, self).__init__(config, train_args, *args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels,64)

    def get_model_lr(self):
        return super(TransformerForPointer, self).get_model_lr() + [
            (self.pointer_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def training_step(self, batch, batch_idx):
        labels: torch.Tensor = batch.pop('labels')
        outputs = self(**batch)
        logits = outputs[0]

        logits = self.pointer_layer(logits, batch['attention_mask'])

        loss = loss_fn(labels, logits)
        f1 = f1_metric(labels, logits)
        self.log_dict({'train_loss': loss, 'f1': f1}, prog_bar=True)
        return loss

