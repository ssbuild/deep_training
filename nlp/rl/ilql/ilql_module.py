# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 11:05

import functools
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import torch
from torch import nn
from torch.nn import functional as F
from .configuration import ILQLConfig
from transformers import PretrainedConfig
from .data_define import ILQLBatch, ILQLSeq2SeqBatch
from ..utils import get_tensor_stats, flatten_dict
from ...models.rl.modeling_ilql import batched_index_select


class ILQLLLMAbstract:
    def forward_llm_value_and_logits(self,input_ids,**kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids,**kwargs)
        logits, qs, target_qs, vs, _ = outputs
        return (logits, qs, target_qs, vs)

class ILQLPrefixLMAbstract:
    def forward_prefix_value_and_logits(self,input_ids,**kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids, **kwargs)
        logits, qs, target_qs, vs, _ = outputs
        return (logits, qs, target_qs, vs)

class ILQLSEQ2SEQAbstract:
    def forward_seq2seq_value_and_logits(self,
                                         input_ids,
                                         attention_mask,
                                         decoder_input_ids,
                                         **kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             decoder_input_ids=decoder_input_ids,
                                             **kwargs)
        logits, qs, target_qs, vs, _, _ = outputs
        return (logits, qs, target_qs, vs)




class ILQLModelLoss(nn.Module, ILQLLLMAbstract, ILQLSEQ2SEQAbstract,ILQLPrefixLMAbstract):

    def forward_ilql_loss(self,batch: dict):
        if self.ilql_config.model_arch_type == "seq2seq":
            batch = ILQLSeq2SeqBatch(**batch)
            logits, qs, target_qs, vs = self.forward_seq2seq_value_and_logits(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
                decoder_input_ids=batch.decoder_input_ids,
            )
        elif self.ilql_config.model_arch_type == "prefixlm":
            batch = ILQLBatch(**batch)
            logits, qs, target_qs, vs = self.forward_prefix_value_and_logits(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
            )
        else:
            batch = ILQLBatch(**batch)
            logits, qs, target_qs, vs = self.forward_llm_value_and_logits(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
            )
        loss,stats = self.loss((logits, (qs, target_qs, vs)), batch)
        return {
            'loss': loss,
            'stats': stats
        }


    def loss(self, outputs, labels):
        logits, (qs, target_qs, vs) = outputs
        terminal_mask = labels.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())
        # check type of labels
        if isinstance(labels, ILQLBatch):
            actions = labels.input_ids[:, 1:].gather(dim=1, index=labels.actions_ixs).unsqueeze(-1)
        else:
            actions = labels.decoder_input_ids[:, 1:].unsqueeze(-1)
        actions = actions.long()
        nactions = actions.shape[1]
        bsize, _, dsize = logits.shape

        Q = [q.gather(-1, actions).squeeze(-1) for q in qs]
        targetQs = [q.gather(-1, actions).squeeze(-1).detach() for q in target_qs]
        targetQ = functools.reduce(torch.minimum, targetQs)

        # The loss_q assumes len(states) == len(rewards) + 1
        # values of current states
        V = vs[:, :-1, 0]
        # values of next states
        Vnext = vs[:, 1:, 0] * labels.dones[:, 1:].to(vs.dtype)
        # target to fit Q
        Q_ = labels.rewards + self.ilql_config.gamma * Vnext.detach()

        loss_qs = [((Qi - Q_) * terminal_mask).pow(2).sum() / n_nonterminal for Qi in Q]
        loss_q = sum(loss_qs)

        targetQ = targetQ.detach()

        loss_v = (
            (
                (targetQ >= V).int() * self.ilql_config.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.ilql_config.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        def cql_loss(q):
            loss = F.cross_entropy(q.reshape(-1, dsize), actions.reshape(-1), reduction="none")
            loss = loss.reshape(bsize, nactions) * terminal_mask
            loss = loss.sum() / n_nonterminal
            return loss

        loss_cql = sum(cql_loss(q) for q in qs)

        # select logits from continuations
        action_logits = batched_index_select(logits, labels.actions_ixs, dim=1)
        cross_entropy = F.cross_entropy(
            action_logits.reshape(-1, dsize),
            actions.reshape(-1),
            reduction="none",
        ).reshape(bsize, nactions)

        with torch.no_grad():
            awac_weight = torch.exp(self.ilql_config.beta * (targetQ - V))

        loss_awac = torch.sum(cross_entropy * awac_weight * terminal_mask) / n_nonterminal
        loss = loss_q + loss_v + self.ilql_config.cql_scale * loss_cql + self.ilql_config.awac_scale * loss_awac

        stats = dict(
            losses=dict(
                loss=loss.item(),
                loss_q=loss_q.item(),
                loss_v=loss_v.item(),
                loss_cql=loss_cql.item(),
                loss_awac=loss_awac.item(),
            ),
            values=get_tensor_stats(V, terminal_mask, n_nonterminal),
            qvalues={str(ix): get_tensor_stats(Q[ix], terminal_mask, n_nonterminal) for ix in range(len(Q))},
            awac_weight=get_tensor_stats(awac_weight, terminal_mask, n_nonterminal),
        )

        return loss, flatten_dict(stats)
