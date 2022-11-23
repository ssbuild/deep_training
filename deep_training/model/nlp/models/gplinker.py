# @Time    : 2022/11/11 20:15
# @Author  : tk
# @FileName: gp_linker.py
import typing

import numpy as np
import torch
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, loss_for_pointer, f1_metric_for_pointer

__all__ = [
    'TransformerForGplinker'
]


def extract_spoes(outputs: typing.List, threshold=1e-8):
    # 抽取subject和object
    pos_logits = outputs[0]
    subjects, objects = set(), set()
    for l, h, t in zip(*np.where(pos_logits > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))

    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            print(outputs[1][:, sh, oh])
            print(outputs)
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]

            print(p1s,p2s)
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((sh,st,p,oh,ot))
    return list(spoes)

class TransformerForGplinker(TransformerModel):
    def __init__(self,with_efficient=False, *args,**kwargs):
        super(TransformerForGplinker, self).__init__(*args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64,RoPE=False, tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels,  64,RoPE=False,tril_mask=False)

    def get_model_lr(self):
        return super(TransformerForGplinker, self).get_model_lr() + [
            (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def compute_loss(self,batch):
        entity_labels: torch.Tensor = batch.pop('entity_labels',None)
        head_labels: torch.Tensor = batch.pop('head_labels',None)
        tail_labels: torch.Tensor = batch.pop('tail_labels',None)
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        logits1 = self.entities_layer(logits, attention_mask)
        logits2 = self.heads_layer(logits, attention_mask)
        logits3 = self.tails_layer(logits, attention_mask)
        if entity_labels is not None:
            loss1 = loss_for_pointer(entity_labels, logits1)
            loss2 = loss_for_pointer(head_labels, logits2)
            loss3 = loss_for_pointer(tail_labels, logits3)
            loss =  (loss1 + loss2 + loss3) / 3
            f1 = (f1_metric_for_pointer(entity_labels, logits1) + f1_metric_for_pointer(head_labels, logits2) + f1_metric_for_pointer(tail_labels, logits3)) / 3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_re1':loss2,
                         'loss_re2':loss3,
                         'f1': f1}
            outputs = (loss_dict,logits1,logits2,logits3,
                       entity_labels,head_labels,tail_labels)

            self.log_dict(loss_dict,prog_bar=True)
        else:
            outputs = (logits1,logits2,logits3)
        return outputs