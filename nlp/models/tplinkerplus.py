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
                "one of ['mix_pooling','mean_pooling','max_pooling','lstm','linear'] "
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

def get_position(pos_val,seqlen, start, end):
    i = math.floor((end + start) / 2)
    j = int((pos_val + i * (i + 1) / 2) - i * seqlen)
    if j >= 0 and j < seqlen:
        return (i, j)
    if j >= seqlen:
        return get_position(pos_val,seqlen, i, end)
    return get_position(pos_val, seqlen,start, i)

def extract_spoes(batch_outputs: typing.List,
                  id2label:dict ,#混合标签,所有标签包括实体和关系
                  rel2id: dict,# 关系标签
                  threshold=1e-8):
    batch_result = []
    seqlen = None
    for shaking_hidden in batch_outputs:
        if seqlen is None:
            seqlen = math.floor(math.sqrt(shaking_hidden.shape[-1] * 2))
        es = []
        es_set = set()
        heads,tails,subs,objs = {},{},{},{}
        for tag_id,pos in zip(*np.where(shaking_hidden > threshold)):
            tag: str = id2label[tag_id]
            tag,tag2 = tag.rsplit('_',1)
            tag2: str = tag2.lower()
            pos = get_position(pos, seqlen, 0, seqlen)
            if tag2 == 'ee':
                es.append((*pos,tag_id))
                es_set.add(pos)
            else:
                tag = rel2id[tag]
                if tag2.startswith('o'):
                    pos = (pos[1],pos[0])
                obj = heads if tag2.endswith('h') else tails
                if tag not in obj:
                    obj[tag] = []
                obj[tag].append(pos)


        for p in heads.keys() & tails.keys():
            for sh,oh in heads[p]:
                for st,ot in tails[p]:
                    s,o = (sh, st), (oh,ot)
                    if s in es_set:
                        if p not in subs:
                            subs[p] = []
                        subs[p].append(s)
                    if o in es_set:
                        if p not in objs:
                            objs[p] = []
                        objs[p].append(o)
        spoes = []
        for p in subs.keys() & objs.keys():
            for s in list(set(subs[p])):
                for o in list(set(objs[p])):
                    spoes.append((s[0] -1,s[1] -1,p,o[0] -1,o[1] -1))
        batch_result.append(spoes)
    return batch_result




def extract_entity(batch_outputs: typing.List,threshold=1e-8):
    batch_result = []

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

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch.get('attention_mask',None)
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits,attention_mask)
        shaking_hiddens = self.fc(shaking_hiddens)
        sampled_tok_pair_indices = None
        if self.tok_pair_sample_rate > 0 and self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
            #             sampled_tok_pair_indices = torch.randint(shaking_hiddens, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1,
                                                                                                    shaking_hiddens.size()[
                                                                                                        -1]))

        shaking_hiddens = torch.transpose(shaking_hiddens, 1, 2)
        if labels is not None:
            if self.training and sampled_tok_pair_indices is not None:
                labels = torch.transpose(labels, 1, 2)
                labels = labels.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1,self.config.num_labels))
                labels = torch.transpose(labels, 1, 2)
            loss,f1 = self.loss_fn(shaking_hiddens, labels,mask=attention_mask,with_matrix=False)
            loss = {
                'loss': loss,
                'f1': f1
            }
            outputs = (loss, shaking_hiddens,labels)
        else:
            shaking_hiddens = torch.transpose(shaking_hiddens, 1, 2)
            outputs = (shaking_hiddens, )
        return outputs

