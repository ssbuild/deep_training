# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 12:26
import copy
import math

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.crf import CRF
from ..layers.mhslayer import MhsLayer
from ..losses.loss_mhslinker import MutiheadlinkerLoss, MutiheadlinkerLossEx

__all__ = [
    'TransformerForMhsLinker'
]
class Chunk:
    l, s, e = -1, -1, -1
    def reset(self):
        self.l = -1
        self.s = -1
        self.e = -1

def get_entities(logits_tags):
    length = len(logits_tags)
    chunks = []
    chunk = Chunk()

    def reset_chunk(chunk: Chunk):
        chunk.reset()

    L = 0
    for indx,T in enumerate(logits_tags):
        if T == 'S':
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            chunk.s = indx
            chunk.e = indx
            chunk.l = L
            chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        elif T == 'B':
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
            chunk.s = indx
            chunk.l = L
        elif T == 'I' and chunk.s != -1:
            if L == chunk.l:
                chunk.e = indx
            else:
                reset_chunk(chunk)
            if indx == length - 1:
                if chunk.e != -1:
                    chunks.append(copy.deepcopy(chunk))
                    reset_chunk(chunk)
        elif T == 'O' and chunk.s != -1:
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        else:
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
    if chunk.e != -1:
        chunks.append(copy.deepcopy(chunk))
        reset_chunk(chunk)

    return [(chunk.l,chunk.s,chunk.e) for chunk in chunks]

def extract_spoes(outputs,threshold = 1e-8):
    id2ents = {
        0: 'O',
        1: 'B',
        2: 'I'
    }
    batch_results = []
    for crf_logits,mhs_logits in zip(outputs[0],outputs[1]):
        ctf_logits = [id2ents[s] for s in crf_logits]
        ents = get_entities(ctf_logits)
        mhs_logits[:,[0, -1]] *= 0
        mhs_logits[:,:,[0, -1]] *= 0
        spoes = []
        if len(ents) > 0:
            e_set = {e[-1]: e for e in ents}
            for p,s,o in zip(*np.where(mhs_logits > threshold)):
                if s not in e_set or o not in e_set:
                    continue
                t1,t2 = e_set[s],e_set[o]
                s = (t1[1], t1[2])
                o = (t2[1], t2[2])
                spoes.append((s[0] -1,s[1] - 1,p,o[0] - 1, o[1] - 1))
        batch_results.append(list(set(spoes)))
    return batch_results


class TransformerForMhsLinker(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForMhsLinker, self).__init__(*args, **kwargs)
        config = self.config

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.crf = CRF(num_tags=3)
        self.mhslayer = MhsLayer(config.hidden_size,config.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = MutiheadlinkerLossEx()

    def get_model_lr(self):
        return super(TransformerForMhsLinker, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.crf, self.config.task_specific_params['learning_rate_for_task']),
            (self.mhslayer, self.config.task_specific_params['learning_rate_for_task']),
            (self.loss_fn, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        seq_labels: torch.Tensor = batch.pop('seq_labels',None)
        mhs_labels: torch.Tensor = batch.pop('mhs_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)

        seq_logits = self.classifier(logits)
        tags = self.crf.decode(seq_logits, attention_mask)
        logits_mhs = self.mhslayer(logits,attention_mask)
        if seq_labels is not None:
            seq_labels = torch.where(seq_labels >= 0, seq_labels, torch.zeros_like(seq_labels))
            loss_crf = self.crf(emissions=seq_logits, tags=seq_labels, mask=attention_mask)
            loss_mhs = self.loss_fn(logits_mhs,mhs_labels)

            if self.training:
                current_step = self.global_step
                total_steps = self.estimated_stepping_batches
                if total_steps == float('inf'):
                    total_steps = 5000
                total_steps *= 0.5
                # z = (2 * self.config.num_labels + 1)
                # w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
                # w_rel = min((self.config.num_labels / z) * current_step / total_steps, (self.config.num_labels / z))
                w_ent = max(0.85 - current_step / total_steps, 0.55)
                w_rel = (1 - w_ent)
                loss = w_ent * loss_crf + w_rel * loss_mhs
                loss_dict = {
                    'crf': loss_crf,
                    'mhs': loss_mhs,
                    'loss': loss
                }
                self.log_dict({
                    'w_ent': w_ent,
                    'w_rel': w_rel
                }, prog_bar=True)
            else:
                loss_dict = None
            outputs = (loss_dict,tags,logits_mhs,seq_labels,mhs_labels)
        else:
            outputs = (tags,logits_mhs,seq_labels,mhs_labels,)
        return outputs

