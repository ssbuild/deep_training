# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 9:27
import math
import typing
from dataclasses import field, dataclass

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.handshakingkernel import HandshakingKernel
from ..losses.loss_tplinker import TplinkerPlusLoss

__all__ = [
    'TransformerForTplinkerPlus'
]

@dataclass
class TplinkerArguments:
    shaking_type: typing.Optional[str] = field(
        default='cln_plus',
        metadata={
            "help": (
                "one of ['cat','cat_plus','cln','cln_plus']"
            )
        },
    )
    inner_enc_type: typing.Optional[str] = field(
        default='lstm',
        metadata={
            "help": (
                "one of ['mix_pooling','mean_pooling','max_pooling','lstm'] "
            )
        },
    )
    tok_pair_sample_rate: typing.Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "(0, 1] How many percent of token paris you want to sample for training, this would slow down the training if set to less than 1. It is only helpful when your GPU memory is not enought for the training. "
            )
        },
    )

def extract_spoes(batch_outputs: typing.List,
                  id2label,#所有标签包括实体和关系
                  rel2id,# 关系id映射
                  threshold=1e-8):
    batch_result = []

    def get_position(pos_val,seqlen, start, end):
        i = math.floor((end + start) / 2)
        j = int((pos_val + i * (i + 1) / 2) - i * seqlen)
        if j >= 0 and j < seqlen:
            return (i, j)
        if j >= seqlen:
            return get_position(pos_val,seqlen, i, end)
        return get_position(pos_val, seqlen,start, i)

    seqlen = None
    for shaking_hidden in batch_outputs:
        if seqlen is None:
            seqlen = math.floor(math.sqrt(shaking_hidden.shape[1] * 2))
        print('seqlen',seqlen,shaking_hidden.shape,np.where(shaking_hidden > 0))
        print(shaking_hidden)
        es = []
        es_set = set()
        sh,oh,st,ot = {},{},{},{}
        for tag_id,pos in zip(*np.where(shaking_hidden > 0)):
            tag: str = id2label[tag_id]
            print(tag)
            tag,tag2 = tag.rsplit('_',1)
            if tag2 == 'EE':
                es.append((*get_position(pos,seqlen,0,seqlen),tag_id))
                es_set.add((es[-1][0],es[-1][1]))
            else:
                obj = eval(tag2.lower())
                if tag not in obj:
                    obj[tag] = []
                obj[tag].append(get_position(pos,seqlen,0,seqlen))

        print('es_set',es_set)
        subs,objs = {},{}
        for k in sh.keys() & st.keys():
            list1 = sh[k]
            list2 = st[k]
            if list1 and list2:
                subs[k] = [(i,j) for i in list1 for j in list2 if (i,j) in es_set]
        for k in oh.keys() & ot.keys():
            list1 = oh[k]
            list2 = ot[k]
            if list1 and list2:
                objs[k] = [(i, j) for i in list1 for j in list2 if (i, j) in es_set]
        spoes = []
        for k in subs.keys() & objs.keys():
            p = rel2id[k]
            for s in subs[k]:
                for o in objs[k]:
                    spoes.append((s[0] -1,s[1] -1,p,o[0] -1,o[1] -1))
        batch_result.append(spoes)
    return batch_result


def extract_entity(batch_outputs: typing.List,threshold=1e-8):
    batch_result = []
    def get_position(pos_val,seqlen, start, end):
        i = math.floor((end + start) / 2)
        j = int((pos_val + i * (i + 1) / 2) - i * seqlen)
        if j >= 0 and j < seqlen:
            return (i, j)
        if j >= seqlen:
            return get_position(pos_val,seqlen, i, end)
        return get_position(pos_val, seqlen,start, i)

    seqlen = None
    for shaking_hidden in batch_outputs:
        if seqlen is None:
            seqlen = math.floor(math.sqrt(shaking_hidden.shape[-1] * 2))
        es = []
        for tag_id,pos in zip(*np.where(shaking_hidden > threshold)):
            start,end = get_position(pos, seqlen, 0, seqlen)
            start -= 1
            end -= 1
            if start < 0 or end < 0:
                continue
            es.append((tag_id,start,end))
        batch_result.append(es)
    return batch_result

# def extract_entity(batch_outputs: typing.List,threshold=1e-8):
#     batch_result = []
#
#     for shaking_hidden in batch_outputs:
#         shaking_hidden[0,-1] *= 0
#         shaking_hidden[:,0,-1] *= 0
#         es = []
#         for tag_id,start,end in zip(*np.where(shaking_hidden > threshold)):
#             start -= 1
#             end -= 1
#             es.append((tag_id,start,end))
#         batch_result.append(es)
#     return batch_result

class TransformerForTplinkerPlus(TransformerModel):
    def __init__(self,  *args, **kwargs):
        tplinker_args = kwargs.pop('tplinker_args',None)
        shaking_type = tplinker_args.shaking_type if tplinker_args else None
        inner_enc_type = tplinker_args.inner_enc_type if tplinker_args else None
        tok_pair_sample_rate = tplinker_args.tok_pair_sample_rate if tplinker_args else 0
        super(TransformerForTplinkerPlus, self).__init__(*args, **kwargs)
        self.tok_pair_sample_rate = tok_pair_sample_rate
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.handshakingkernel = HandshakingKernel(self.config.hidden_size,shaking_type,inner_enc_type)
        self.fc = nn.Linear(self.config.hidden_size,self.config.num_labels)
        self.loss_fn = TplinkerPlusLoss()

    def get_model_lr(self):
        return super(TransformerForTplinkerPlus, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.handshakingkernel, self.config.task_specific_params['learning_rate_for_task']),
            (self.fc, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, batch,batch_idx):
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch.get('attention_mask',None)
        outputs = self(**batch)
        logits = outputs[0]
        if self.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits,attention_mask)
        if self.training:
            shaking_hiddens = self.dropout(shaking_hiddens)
        shaking_hiddens = self.fc(shaking_hiddens)
        shaking_hiddens = torch.transpose(shaking_hiddens,1,2)
        # mask = torch.tril(torch.ones_like(shaking_hiddens,dtype=torch.float32),1)
        # shaking_hiddens = shaking_hiddens * mask
        if labels is not None:
            loss = self.loss_fn(shaking_hiddens, labels)
            outputs = (loss, shaking_hiddens,labels)
        else:
            outputs = (shaking_hiddens, )
        return outputs