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
        default='mix_pooling',
        metadata={
            "help": (
                "one of ['mix_pooling','lstm'] "
            )
        },
    )
    tok_pair_sample_rate: typing.Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "tok_pair_sample_rate "
            )
        },
    )

def extract_spoes(batch_outputs: typing.List,
                  id2label,#所有标签包括实体和关系
                  rel2id,# 关系id映射
                  threshold=1e-8):
    batch_result = []
    seq_map = None
    for shaking_hidden in batch_outputs:
        seqlen = len(batch_outputs)
        if seq_map is None:
            seq_map = {}
            get_pos = lambda x0, x1: x0 * seqlen + x1 - x0 * (x0 + 1) // 2
            for i in range(seqlen):
                for j in range(i, seqlen):
                    seq_map[get_pos(i, j)] = (i, j)
        es = []
        es_set = set()
        sh,oh,st,ot = {},{},{},{}
        for pos,tag_id in zip(np.where(shaking_hidden >threshold)):
            tag: str = id2label[tag_id]
            tag,tag2 = tag.rsplit('_',2)
            if tag2 == 'EE':
                es.append((*seq_map[pos],tag_id))
                es_set.add(seq_map[pos])
            else:
                obj = eval(tag2.lower())
                if tag not in obj:
                    obj[tag] = []
                obj[tag].append(seq_map[pos])
        subs,objs = {},{}
        for k,list1 in sh.items():
            list2 = st.get(k,[])
            subs[k] = [(i,j) for i in list1 for j in list2 if (i,j) in es_set]
        for k, list1 in oh.items():
            list2 = ot.get(k, [])
            objs[k] = [(i, j) for i in list1 for j in list2 if (i, j) in es_set]
        spoes = []
        for k in subs.keys() & objs.keys():
            p = rel2id[k]
            for s in subs[k]:
                for o in objs[k]:
                    spoes.append((s[0] -1,s[1] -1,p,o[0] -1,o[1] -1))

        batch_result.append(spoes)
    return batch_result


class TransformerForTplinkerPlus(TransformerModel):
    def __init__(self,  *args, **kwargs):
        tplinker_args = kwargs.pop('tplinker_args',None)
        shaking_type = tplinker_args.shaking_type if tplinker_args else None
        inner_enc_type = tplinker_args.inner_enc_type if tplinker_args else None
        tok_pair_sample_rate = tplinker_args.tok_pair_sample_rate if tplinker_args else None
        super(TransformerForTplinkerPlus, self).__init__(*args, **kwargs)
        self.tok_pair_sample_rate = tok_pair_sample_rate
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.handshakingkernel = HandshakingKernel(self.config.hidden_size,shaking_type,inner_enc_type)
        self.fc = nn.Linear(self.config.hidden_size,self.config.num_labels)
        self.loss_fn = TplinkerPlusLoss()

    def get_model_lr(self):
        return super(TransformerForTplinkerPlus, self).get_model_lr() + [
            (self.handshakingkernel, self.config.task_specific_params['learning_rate_for_task']),
            (self.fc, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, batch,batch_idx):
        labels: torch.Tensor = batch.pop('labels', None)

        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits)

        sampled_tok_pair_indices = None
        if self.training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0],
                                                                                        1)
            #             sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(shaking_hiddens.device)

            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1,
                                                                                                    shaking_hiddens.size()[
                                                                                                        -1]))


        shaking_hiddens = self.fc(shaking_hiddens)
        if labels is not None:
            # sampled_tok_pair_indices: (batch_size, ~segment_len)
            # batch_small_shaking_tag: (batch_size, ~segment_len, tag_size)
            if self.training:
                batch_small_shaking_tag = labels.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, self.config.num_labels))
                loss = self.loss_fn(shaking_hiddens, batch_small_shaking_tag)
            else:
                loss = None
            outputs = (loss, shaking_hiddens,labels)
        else:
            outputs = (shaking_hiddens, )
        return outputs