# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import torch
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer,loss_fn,f1_metric

__all__ = [
    'TransformerForPointer'
]

class TransformerForPointer(TransformerModel):
    def __init__(self,with_efficient=True, *args,**kwargs):
        super(TransformerForPointer, self).__init__(*args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels,64)

    def get_model_lr(self):
        return super(TransformerForPointer, self).get_model_lr() + [
            (self.pointer_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self,batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        outputs = self(**batch)
        logits = self.pointer_layer(outputs[0], batch['attention_mask'])
        if labels is not None:
            loss = loss_fn(labels, logits)
            f1 = f1_metric(labels, logits)
            loss_dict = {'loss': loss, 'f1': f1}
            outputs = (loss_dict,logits,labels)
        else:
            outputs = (logits,)
        return outputs

