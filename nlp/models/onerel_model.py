# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 12:26
#reference: https://github.com/ssnvxia/OneRel

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel

__all__ = [
    'TransformerForOneRel'
]


def extract_spoes(outputs):
    batch_result = []
    for logits in outputs:
        #rel,s,s
        hs,hts,ts={},{},{}
        for obj,filter in [(hs,1),(hts,2),(ts,3)]:
            for p, p1, p2 in zip(*np.where(logits == filter)):
                if p not in obj:
                    obj[p] = []
                obj[p].append((p1,p2))
        spoes = set()
        for p in hs.keys() & ts.keys() & hts.keys():
            ht_list = hts[p]
            for sh,oh in hs[p]:
                for st,ot in ts[p]:
                    if sh <= st and oh <= ot:
                        if (sh,ot) in ht_list:
                            spoes.add((sh-1,st-1,p,oh-1 ,ot-1))
        batch_result.append(list(spoes))
    return batch_result


class TransformerForOneRel(TransformerModel):
    def __init__(self, *args,**kwargs):
        entity_pair_dropout = kwargs.pop('entity_pair_dropout',0.1)

        super(TransformerForOneRel, self).__init__(*args,**kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.tag_size = 4
        self.projection_matrix_layer = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 3)
        self.relation_matrix_layer = nn.Linear(self.config.hidden_size * 3, self.config.num_labels * self.tag_size)

        self.dropout_2 = nn.Dropout(entity_pair_dropout)
        # self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none',ignore_index=-100)


    def get_model_lr(self):
        return super(TransformerForOneRel, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate']),
            (self.dropout_2, self.config.task_specific_params['learning_rate']),
            (self.relation_matrix_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.projection_matrix_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def _triple_score_matrix(self, seq_output):
        # encoded_text: [batch_size, seq_len, bert_dim(768)] 1,2,3
        batch_size, seq_len, bert_dim = seq_output.size()
        # head: [batch_size, seq_len * seq_len, bert_dim(768)] 1,1,1, 2,2,2, 3,3,3
        head_representation = seq_output.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(
            batch_size, seq_len * seq_len, bert_dim)
        # tail: [batch_size, seq_len * seq_len, bert_dim(768)] 1,2,3, 1,2,3, 1,2,3
        tail_representation = seq_output.repeat(1, seq_len, 1)
        # [batch_size, seq_len * seq_len, bert_dim(768)*2]
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)

        # [batch_size, seq_len * seq_len, bert_dim(768)*3]
        entity_pairs = self.projection_matrix_layer(entity_pairs)

        entity_pairs = self.dropout_2(entity_pairs)

        # entity_pairs = self.activation(entity_pairs)

        # [batch_size, seq_len * seq_len, rel_num * tag_size] -> [batch_size, seq_len, seq_len, rel_num, tag_size]
        triple_scores = self.relation_matrix_layer(entity_pairs).reshape(batch_size, seq_len, seq_len, self.config.num_labels,
                                                                         self.tag_size)

        #[b, s, s, rel_num, tag_size]
        return triple_scores

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        # [b, s, s, rel_num, tag_size]
        logits = self._triple_score_matrix(logits)
        tags = torch.argmax(logits,dim=-1).permute(0,3,1,2)
        if labels is not None:
            #b,n,s,s
            loss = self.loss_fn(logits.permute(0, 4, 3, 1, 2), labels)
            loss = loss.mean(-1).mean(-1).sum() / logits.size(0)
            # masks = torch.zeros_like(loss,dtype=torch.float,device=logits.device)
            # for m,l in zip(masks,torch.sum(attention_mask,dim=-1)):
            #     m[:,:l,:l] = 1.0
            # loss = torch.sum(loss * masks) / torch.sum(masks)
            outputs = (loss,tags,labels)
        else:
            outputs = (tags,)
        return outputs

