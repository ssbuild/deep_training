# @Time    : 2022/12/10 0:44
# @Author  : tk
# @FileName: w2ner.py
from typing import Union, List

import numpy as np
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from .transformer import TransformerModel
__all__ = [
    'TransformerForSpanNer'
]

from ..metrics.pointer import metric_for_pointer

def extract_lse_singlelabel(outputs,top_n=1):
    if top_n is not None and top_n <= 0:
        top_n = -1
    batch_result = []
    for heads,tails in zip(outputs[0].argmax(-1),outputs[1].argmax(-1)):
        heads[[0, -1]] *= 0
        tails[[0, -1]] *= 0
        pts = []
        for h, l1 in enumerate(heads):
            if l1 == 0:
                continue
            n = 0
            for t,l2 in enumerate(tails):
                if l1 != l2 or h>t:
                    continue
                pts.append((l1 - 1, h - 1, t - 1))
                n += 1
                if n >= top_n:
                    break
        batch_result.append(pts)
    return batch_result


def extract_lse_mutilabel(outputs,threshold=0.5,top_n=1):
    if top_n is not None and top_n <= 0:
        top_n = -1
    batch_result = []
    for logits in outputs:
        hs,ts = [],[]
        logits[[0,-1]] *= 0
        for pos,l,ht in zip(*np.where(logits > threshold)):
            obj = hs if ht == 0 else ts
            obj.append((l,pos))
        pts = []
        for l1, h in hs:
            n = 0
            for l2, t in ts:
                if l1 != l2 or h > t:
                    continue
                pts.append((l1, h - 1, t - 1))
                n += 1
                if n >= top_n:
                    break
        batch_result.append(pts)
    return batch_result


class TransformerForSpanNer(TransformerModel):
    def __init__(self, *args,**kwargs):
        with_mutilabel = kwargs.pop('with_mutilabel',True)
        super(TransformerForSpanNer, self).__init__(*args,**kwargs)
        self.with_mutilabel = with_mutilabel
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.with_mutilabel:
            self.span_layer = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_labels * 2),
                nn.Sigmoid()
            )
            self.loss_fn = nn.BCELoss(reduction='none')

            self.compute_loss = self.compute_loss_for_mutilabel
        else:
            self.span_layer = nn.Sequential(
                nn.Linear(config.hidden_size, (config.num_labels + 1) * 2),
            )
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            self.compute_loss = self.compute_loss_for_singlelabel



    def get_model_lr(self):
        return super(TransformerForSpanNer, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.span_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def compute_loss_for_mutilabel(self,*args,**batch):
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = torch.reshape(self.span_layer(logits), shape=logits.shape[:2] + (-1, 2))
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            attention_mask = attention_mask.float()
            loss = (loss.mean(-1).sum(-1) * attention_mask.float()).sum() / attention_mask.sum()
            outputs = (loss, logits, labels)
        else:
            outputs = (logits,)
        return outputs

    def compute_loss_for_singlelabel(self, *args,**batch):
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)

        logits = torch.reshape(self.span_layer(logits), shape=logits.shape[:2] + (-1, 2))
        heads_logits,tails_logits = torch.unbind(logits, dim=-1)

        mask = attention_mask.float().unsqueeze(2).expand(-1,-1,heads_logits.size()[-1])
        if labels is not None:
            heads_labels, tails_labels = torch.unbind(labels,  dim=-1)
            loss1 = self.loss_fn(torch.transpose(heads_logits,2,1), heads_labels)
            loss2 = self.loss_fn(torch.transpose(tails_logits,2,1), tails_labels)
            attention_mask = attention_mask.float()
            loss1 = (loss1 * attention_mask).sum() / attention_mask.sum()
            loss2 = (loss2 * attention_mask).sum() / attention_mask.sum()

            loss = {
                'heads_loss': loss1,
                'tails_loss': loss2,
                'loss': (loss1 + loss2) / 2
            }
            outputs = (loss,
                       heads_logits.softmax(-1) *mask,
                       tails_logits.softmax(-1)* mask,
                       labels)
        else:
            outputs = (heads_logits.softmax(-1) * mask,
                       tails_logits.softmax(-1) * mask,)
        return outputs

    # def compute_loss(self, *args,**batch) -> tuple:
    #     ...

    # def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
    #     label2id = self.config.label2id
    #     threshold = 0.5
    #     preds, trues = [], []
    #     eval_labels = self.eval_labels
    #     for i, o in enumerate(outputs):
    #         logits, _ = o['outputs']
    #         preds.extend(extract_lse(logits, threshold))
    #         bs = len(logits)
    #         trues.extend(eval_labels[i * bs: (i + 1) * bs])
    #
    #     f1, str_report = metric_for_pointer(trues, preds, label2id)
    #     print(f1)
    #     print(str_report)
    #     self.log('val_f1', f1, prog_bar=True)