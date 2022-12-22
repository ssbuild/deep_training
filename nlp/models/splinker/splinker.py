# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 17:15
import numpy as np
import torch
from torch import nn
from ..transformer import TransformerModel
from ...losses.loss_splinker import BCELossForSplinker

__all__ = [
    'TransformerForSplinker'
]

def extract_spoes(logits_all, seq_len_all, id2labels):
    batch_result = []
    num_labels = len(id2labels)
    for (i, (logits, seq_len)) in enumerate(zip(logits_all, seq_len_all)):
        logits = logits[1:seq_len + 1]
        logits = np.transpose(logits, (1, 0))
        s_objs,o_objs = set(),set()
        for p in np.argwhere(logits > 0.5):
            if p[0] == 0 or p[0] == 1:
                continue
            objs = s_objs if p[0] < num_labels + 2 else o_objs
            objs.add((p[0], p[1]))
        spoes = []
        for s in s_objs:
            for o in o_objs:
                if s[0] + num_labels == o[0]:
                    spoes.append([s[1],s[1],s[0],o[1],o[1]])
        spoes_ = []
        for sh,st,p,oh,ot in spoes:
            for j in range(sh + 1,seq_len):
                if logits[1][j] <= 0.5:
                    break
                st += 1
            for j in range(ot + 1, seq_len):
                if logits[1][j] <= 0.5:
                    break
                ot += 1
            spoes_.append((sh,st,p - 2,oh,ot))
        batch_result.append(spoes_)
    return batch_result




class TransformerForSplinker(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(TransformerForSplinker, self).__init__(*args, **kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels * 2 +2)
        self.loss_fn = BCELossForSplinker()
        self.sigmoid = nn.Sigmoid()

    def get_model_lr(self):
        return super(TransformerForSplinker, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
        ]
    

    def compute_loss(self,*args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        mask = batch.pop('mask')
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        logits = self.dropout(logits)
        logits = self.sigmoid(self.classifier(logits))
        tags = torch.where(logits > 0.5, torch.ones_like(logits,dtype=torch.int32), torch.zeros_like(logits,dtype=torch.int32))
        seqlen = torch.sum(attention_mask,dim=1,keepdim=False).long() -2
        if labels is not None:
            loss = self.loss_fn(logits=logits, labels=labels, mask=mask)
            outputs = (loss,tags,seqlen,labels)
        else:
            outputs = (tags,seqlen,)
        return outputs
