# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
from typing import Union, List

import numpy as np
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer,loss_fn,f1_metric

__all__ = [
    'TransformerForPointer'
]

from ..metrics.pointer import metric_for_pointer


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

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        id2label = self.config.id2label
        threshold = 1e-8
        preds, trues = [], []
        for o in outputs:
            logits, label = o['outputs']
            logits[:, :, [0, -1]] -= np.inf
            logits[:, :, :, [0, -1]] -= np.inf
            assert len(logits) == len(label)
            for p, t in zip(logits, label):
                a_result = []
                for (l, s, e) in zip(*np.where(p > threshold)):
                    a_result.append((l, s, e))
                preds.append(a_result)
                b_result = []
                for (l, s, e) in zip(*np.where(t > threshold)):
                    b_result.append((l, s, e))
                trues.append(b_result)
        f1, str_report = metric_for_pointer(trues, preds, id2label)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)
