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


def extract_spoes(outputs):
    ents: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    batch_result = []
    seq_map = None
    for ents, heads, tails in zip(outputs[0].argmax(-1),outputs[1].argmax(-1),outputs[2].argmax(-1)):
        seqlen = ents.shape[0]
        if seq_map is None:
            seq_map = {}
            get_pos = lambda x0, x1: x0 * seqlen + x1 - x0 * (x0 + 1) // 2
            for i in range(seqlen):
                for j in range(i,seqlen):
                    seq_map[get_pos(i,j)] = (i,j)
        e_map = set()
        for e in zip(*ents.nonzero()):
            if e not in seq_map:
                continue
            e_map.add(seq_map[e])

        spoes = []
        #num,s
        for p1,h in zip(*heads.nonzero()):
            tagid1 = heads[p1,h]
            for p2, t in zip(*tails.nonzero()):
                tagid2 = tails[p2, t]
                if p1 != p2 or tagid1 == tagid2:
                    continue
                if h not in seq_map or t not in seq_map:
                    continue
                h = seq_map[h]
                t = seq_map[t]
                pt1 = (h[0], t[0])
                pt2 = (h[1], t[1])
                if pt1 not in e_map or pt2 not in e_map:
                    continue
                s,o= (pt1,pt2) if tagid1 == 1 else (pt2,pt1)
                spoes.append((s[0]-1,s[1]-1,p1,o[0]-1,o[1]-1))
        batch_result.append(spoes)
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
        self.head_rel_fc_list = [nn.Linear(self.config.hidden_size, 3) for _ in range(self.config.num_labels)]
        self.tail_rel_fc_list = [nn.Linear(self.config.hidden_size, 3) for _ in range(self.config.num_labels)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
        self.loss_fn = TplinkerLoss()

    def get_model_lr(self):
        return super(TransformerForTplinker, self).get_model_lr() + [
            (self.handshakingkernel, self.config.task_specific_params['learning_rate_for_task']),
            (self.ent_fc, self.config.task_specific_params['learning_rate_for_task']),

        ] + \
        list((layer, self.config.task_specific_params['learning_rate_for_task']) for layer in self.head_rel_fc_list) + \
        list((layer, self.config.task_specific_params['learning_rate_for_task']) for layer in self.tail_rel_fc_list)

    def compute_loss(self, batch,batch_idx):
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        shaking_hiddens = self.handshakingkernel(logits)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size > 0:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = attention_mask.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        #         if self.dist_emb_size != -1 and self.ent_add_dist:
        #             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
        #         else:
        #             shaking_hiddens4ent = shaking_hiddens
        #         if self.dist_emb_size != -1 and self.rel_add_dist:
        #             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
        #         else:
        #             shaking_hiddens4rel = shaking_hiddens

        # b,s*(s+1)/2,3
        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))


        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        if entity_labels is not None:
            z = (2 * self.config.num_labels + 1)
            current_step = self.global_step

            total_steps = self.estimated_stepping_batches
            if total_steps == float('inf'):
                total_steps = 5000

            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((self.config.num_labels / z) * current_step / total_steps, (self.config.num_labels / z))

            loss1 = self.loss_fn(ent_shaking_outputs, entity_labels)
            loss2 = self.loss_fn(head_rel_shaking_outputs, head_labels)
            loss3 = self.loss_fn(tail_rel_shaking_outputs, tail_labels)
            loss = w_ent * loss1 + w_rel * loss2 + w_rel * loss3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_head': loss2,
                         'loss_tail': loss3}
            outputs = (loss_dict, ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs,
                       entity_labels, head_labels, tail_labels)
        else:
            outputs = (ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs)
        return outputs
