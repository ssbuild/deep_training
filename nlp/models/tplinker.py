# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 9:27
import typing
from dataclasses import dataclass, field

import numpy as np
import math
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.handshakingkernel import HandshakingKernel
from ..losses.loss_tplinker import TplinkerLoss
__all__ = [
    'TransformerForTplinker'
]

@dataclass
class TplinkerArguments:
    shaking_type: typing.Optional[str] = field(
        default='cat',
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
    dist_emb_size: typing.Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "dist_emb_size"
            )
        },
    )
    ent_add_dist: typing.Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "ent_add_dist "
            )
        },
    )
    rel_add_dist: typing.Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "rel_add_dist "
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

def extract_spoes(outputs):
    batch_result = []
    seqlen = None
    for ents, heads, tails in zip(outputs[0].argmax(-1),
                                  outputs[1].argmax(-1),
                                  outputs[2].argmax(-1)):
        ents[[0, -1]] *= 0
        heads[:, [0, -1]] *= 0
        tails[:, [0, -1]] *= 0

        if seqlen is None:
            seqlen = math.floor(math.sqrt(ents.shape[0] * 2))

        e_map = set()
        for e in ents.nonzero()[0]:
            e_map.add(get_position(e,seqlen,0,seqlen))
        spoes = []
        if len(e_map) > 0:
            hs ,ts = {}, {}
            for tagid1,p1 in zip(*heads.nonzero()):
                flag = False
                for tagid2,p2 in zip(*tails.nonzero()):
                    if tagid1 != tagid2:
                        continue
                    if not flag:
                        flag = True
                        if tagid1 not in hs:
                            hs[tagid1] = []
                        hs[tagid1].append((p1,heads[tagid1,p1]))
                    if tagid2 not in ts:
                        ts[tagid2] = []
                    ts[tagid2].append((p2, tails[tagid2, p2]))

            for p in hs.keys() & ts.keys():
                for h,type1 in hs[p]:
                    h = get_position(h, seqlen, 0, seqlen)
                    if type1 == 2:
                        h = (h[1],h[0])
                    for t,type2 in ts[p]:
                        t = get_position(t, seqlen, 0, seqlen)
                        if type2 == 2:
                            t = (t[1], t[0])
                        s = (h[0], t[0])
                        o = (h[1], t[1])
                        if s not in e_map or o not in e_map:
                            continue
                        spoes.append((s[0] - 1, s[1] - 1, p, o[0] - 1, o[1] - 1))

        batch_result.append(list(set(spoes)))
    return batch_result

class TransformerForTplinker(TransformerModel):
    def __init__(self,  *args, **kwargs):
        tplinker_args = kwargs.pop('tplinker_args',None)
        shaking_type = tplinker_args.shaking_type if tplinker_args else None
        inner_enc_type = tplinker_args.inner_enc_type if tplinker_args else None

        dist_emb_size = tplinker_args.dist_emb_size if tplinker_args else -1
        ent_add_dist = tplinker_args.ent_add_dist if tplinker_args else -1
        rel_add_dist = tplinker_args.rel_add_dist if tplinker_args else -1
        super(TransformerForTplinker, self).__init__(*args, **kwargs)

        self.dist_emb_size = dist_emb_size
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.handshakingkernel = HandshakingKernel(self.config.hidden_size,shaking_type,inner_enc_type)

        self.ent_fc = nn.Linear(self.config.hidden_size, 2)
        self.heads_layer = nn.Linear(self.config.hidden_size, self.config.num_labels * 3)
        self.tails_layer = nn.Linear(self.config.hidden_size, self.config.num_labels * 3)
        self.loss_ent_fn = TplinkerLoss(reduction='sum')
        self.loss_rel_fn = TplinkerLoss(reduction='sum')

    def get_model_lr(self):
        return super(TransformerForTplinker, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.handshakingkernel, self.config.task_specific_params['learning_rate_for_task']),
            (self.ent_fc, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_ent_fn, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_rel_fn, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def seq_masking(self,logits,mask,seq_dim,value=-100):
        lens = torch.sum(mask,-1)
        seqlen = math.floor(math.sqrt(logits.size()[seq_dim] * 2))
        get_virtual_pos = lambda x0, x1: x0 * seqlen + x1 - int(x0 * (x0 + 1) / 2)
        bs = len(logits)
        for i in range(bs):
            ilen = lens[i]
            ilogits = logits[i]
            s = get_virtual_pos(ilen,ilen)
            e = get_virtual_pos(seqlen - 1,seqlen -1) + 1
            if s < e:
                if seq_dim == 2:
                    ilogits[:,s:e] = value
                else:
                    ilogits[s:e] = value
        return logits


    def compute_loss(self, *args,**batch) -> tuple:
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits,attention_mask)

        # add distance embeddings if it is set
        # if self.dist_emb_size > 0:
        #     # set self.dist_embbedings
        #     hidden_size = shaking_hiddens.size()[-1]
        #     if self.dist_embbedings is None:
        #         dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
        #         for d in range(self.dist_emb_size):
        #             for i in range(hidden_size):
        #                 if i % 2 == 0:
        #                     dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
        #                 else:
        #                     dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
        #         seq_len = attention_mask.size()[1]
        #         dist_embbeding_segs = []
        #         for after_num in range(seq_len, 0, -1):
        #             dist_embbeding_segs.append(dist_emb[:after_num, :])
        #         self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)
        #
        #     if self.ent_add_dist:
        #         shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
        #             shaking_hiddens.size()[0], 1, 1)
        #     if self.rel_add_dist:
        #         shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
        #             shaking_hiddens.size()[0], 1, 1)
        # else:
        #     shaking_hiddens4ent = shaking_hiddens
        #     shaking_hiddens4rel = shaking_hiddens

        # b,s,3
        ent_shaking_outputs = self.ent_fc(shaking_hiddens)
        head_rel_shaking_outputs = self.heads_layer(shaking_hiddens)
        tail_rel_shaking_outputs = self.tails_layer(shaking_hiddens)
        head_rel_shaking_outputs = torch.reshape(head_rel_shaking_outputs,head_rel_shaking_outputs.shape[:2] + (-1, 3)).transpose(1,2)
        tail_rel_shaking_outputs = torch.reshape(tail_rel_shaking_outputs, tail_rel_shaking_outputs.shape[:2] + (-1, 3)).transpose(1,2)

        ent_shaking_outputs = self.seq_masking(ent_shaking_outputs,attention_mask,1,-100000)
        head_rel_shaking_outputs = self.seq_masking(head_rel_shaking_outputs, attention_mask,2,-100000)
        tail_rel_shaking_outputs = self.seq_masking(tail_rel_shaking_outputs, attention_mask,2,-100000)
        if entity_labels is not None:
            entity_labels = self.seq_masking(entity_labels,attention_mask,1)
            head_labels = self.seq_masking(head_labels, attention_mask,2)
            tail_labels = self.seq_masking(tail_labels, attention_mask,2)

            if self.training:
                current_step = self.global_step
                total_steps = self.estimated_stepping_batches
                if total_steps == float('inf'):
                    total_steps = 5000
                total_steps *= 0.5
                # z = (2 * self.config.num_labels + 1)
                # w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
                # w_rel = min((self.config.num_labels / z) * current_step / total_steps, (self.config.num_labels / z))
                w_ent = max(0.85  - current_step / total_steps, 0.4)
                w_rel = (1 - w_ent ) / 2.
                loss1 = self.loss_ent_fn(ent_shaking_outputs, entity_labels)
                loss2 = self.loss_rel_fn(head_rel_shaking_outputs, head_labels)
                loss3 = self.loss_rel_fn(tail_rel_shaking_outputs, tail_labels)
                weight = 1.0 / ent_shaking_outputs.size()[0]
                loss1 *= weight
                loss2 *= weight
                loss3 *= weight
                loss = w_ent * loss1  + w_rel * loss2  + w_rel * loss3
                self.log_dict({
                    'w_ent': w_ent,
                    'w_rel':w_rel
                    },prog_bar=True)
                loss_dict = {'loss': loss,
                             'loss_entities': loss1,
                             'loss_head': loss2,
                             'loss_tail': loss3}
            else:
                loss_dict = None
            outputs = (loss_dict, ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs,
                       entity_labels, head_labels, tail_labels)
        else:
            outputs = (ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs)
        return outputs
