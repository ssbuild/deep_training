# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
from typing import Union, List

import numpy as np
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn

from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, f1_metric_for_pointer
from ..losses.loss_globalpointer import loss_for_pointer
from ..metrics.pointer import metric_for_pointer

__all__ = [
    'TransformerForPointer'
]




class TransformerForPointer(TransformerModel):
    def __init__(self,eval_labels,with_efficient=True, *args,**kwargs):
        super(TransformerForPointer, self).__init__(*args, **kwargs)

        self.eval_labels = eval_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels,64)

    def get_model_lr(self):
        return super(TransformerForPointer, self).get_model_lr() + [
            (self.pointer_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self,batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        outputs = self(**batch)
        logits = self.dropout(outputs[0])
        logits = self.pointer_layer(logits, batch['attention_mask'])
        if labels is not None:
            loss = loss_for_pointer(labels, logits)
            f1 = f1_metric_for_pointer(labels, logits)
            loss_dict = {'loss': loss, 'f1': f1}
            outputs = (loss_dict,logits,labels)
        else:
            outputs = (logits,)
        return outputs

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        label2id = self.config.label2id
        threshold = 1e-7
        preds, trues = [], []

        index = 0
        for o in outputs:
            logits, label = o['outputs']
            for p, t in zip(logits, label):
                a_result = []
                for (l, s, e) in zip(*np.where(p > threshold)):
                    a_result.append((l, s, e))
                preds.append(a_result)


                print(index)

                b_result = []

                for (l, s, e) in self.eval_labels[index]:
                    b_result.append((l, s, e))
                index +=1
                # b_result = []
                # for (l, s, e) in zip(*np.where(t > threshold)):
                #     b_result.append((l, s, e))
                # trues.append(b_result)
        f1, str_report = metric_for_pointer(trues, preds, label2id)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)
