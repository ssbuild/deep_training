# @Time    : 2022/11/11 20:15
# @Author  : tk
# @FileName: gp_linker.py
import typing

import numpy as np
import torch
from torch import nn

from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer
from ..losses.loss_globalpointer import loss_for_gplinker

__all__ = [
    'TransformerForGplinker'
]


def extract_spoes_from_labels(outputs: typing.List):
    batch_spoes = []
    for l1, l2, l3 in zip(outputs[0], outputs[1], outputs[2]):
        subjects, objects = set(), set()
        for h in l1[0]:
            if h[0] != 0 and h[1] != 0:
                subjects.add((h[0], h[1]))
    
        for t in l1[1]:
            if t[0] != 0 and t[1] != 0:
                objects.add((t[0], t[1]))
    
        spoes = set()
        for p,(hs,ts) in enumerate(zip(l2,l3)):
            for h,t in zip(hs,ts):
                h = tuple(h.tolist())
                t = tuple(t.tolist())
                if h in subjects and t in objects:
                    spoes.add((h[0], h[1], p, t[0], t[1]))
        batch_spoes.append(list(spoes))
    return batch_spoes

def extract_spoes(outputs: typing.List, threshold=1e-8):
    batch_spoes = []
    for logit1,logit2,logit3 in zip(outputs[0],outputs[1],outputs[2]):
        # 抽取subject和object
        subjects, objects = set(), set()
        logit1[:, [0, -1]] -= np.inf
        logit1[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(logit1 > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))

        # 识别对应的predicate
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(logit2[:, sh, oh] > threshold)[0]
                p2s = np.where(logit3[:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((sh, st, p, oh, ot))
        batch_spoes.append(list(spoes))
    return batch_spoes




class TransformerForGplinker(TransformerModel):
    def __init__(self, with_efficient=False, *args, **kwargs):
        super(TransformerForGplinker, self).__init__(*args, **kwargs)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64, RoPE=False,
                                              tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64, RoPE=False,
                                              tril_mask=False)

    def get_model_lr(self):
        return super(TransformerForGplinker, self).get_model_lr() + [
            (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, batch):
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = self.dropout(outputs[0])
        logits1 = self.entities_layer(logits, attention_mask)
        logits2 = self.heads_layer(logits, attention_mask)
        logits3 = self.tails_layer(logits, attention_mask)
        if entity_labels is not None:
            loss1 = loss_for_gplinker(entity_labels, logits1)
            loss2 = loss_for_gplinker(head_labels, logits2)
            loss3 = loss_for_gplinker(tail_labels, logits3)
            loss = (loss1 + loss2 + loss3) / 3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_head': loss2,
                         'loss_tail': loss3}
            outputs = (loss_dict, logits1, logits2, logits3,
                       entity_labels, head_labels, tail_labels)

            self.log_dict(loss_dict, prog_bar=True)
        else:
            outputs = (logits1, logits2, logits3)
        return outputs
