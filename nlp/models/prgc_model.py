# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 13:41
#reference: https://github.com/hy-struggle/PRGC
import copy
import typing
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel

__all__ = [
    'TransformerForPRGC'
]

@dataclass
class PrgcModelArguments:
    dropout: typing.Optional[int] = field(
        default=0.1,
        metadata={
            "help": (
                "dropout of the task"
            )
        },
    )
    corres_threshold: typing.Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "threshold of global correspondence"
            )
        },
    )
    rel_threshold: typing.Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "threshold of global judgement"
            )
        },
    )
    ensure_corres: typing.Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "correspondence ablation"
            )
        },
    )
    ensure_rel: typing.Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "relation judgement ablation"
            )
        },
    )
    emb_fusion: typing.Optional[str] = field(
        default='concat',
        metadata={
            "help": (
                "way to embedding,one of [concat,sum]"
            )
        },
    )


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

    return [(chunk.s,chunk.e) for chunk in chunks]


def extract_spoes(outputs: typing.List,rel_threshold = 0.5,corres_threshold=0.5):
    id2ents = {
        0: 'O',
        1: 'B',
        2: 'I'
    }
    batch_result = []
    #b,num_rels ; b,n,2,s,tags , b,s,s
    if outputs[2] is None:
        iter_objs = zip(outputs[0], outputs[1])
    else:
        iter_objs = zip(outputs[0], outputs[1], outputs[2])

    item: tuple
    for item in iter_objs:
        if len(item) == 2:
            item = item + (None,)
        pred_rels, pred_seqs, pred_corres = item
        potential_rels = []
        for rel_id,pred_rel in enumerate(pred_rels):
            if pred_rel > rel_threshold:
                potential_rels.append(rel_id)

        spoes = set()
        if potential_rels:
            if pred_corres is not None:
                sohs = set()
                for h1,h2 in zip(*np.where(pred_corres > corres_threshold)):
                    sohs.add((h1,h2))
            else:
                sohs = None

            for p,pred_seq in zip(potential_rels,pred_seqs):
                sub_preds = np.argmax(pred_seq[0], axis=-1)
                obj_preds = np.argmax(pred_seq[1], axis=-1)
                subs = get_entities([id2ents[_] for _ in sub_preds])
                objs = get_entities([id2ents[_] for _ in obj_preds])
                for sh,st in subs:
                    for oh,ot in objs:
                        if sohs is not None:
                            if (sh,oh) not in sohs:
                                continue
                        spoes.add((sh-1,st-1,p,oh-1,ot-1))
        batch_result.append(list(spoes))
    return batch_result

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output

class TransformerForPRGC(TransformerModel):
    def __init__(self, *args,**kwargs):
        prgcmodel_args: PrgcModelArguments = kwargs.pop('prgcmodel_args')
        super(TransformerForPRGC, self).__init__(*args,**kwargs)
        self.prgcmodel_args = prgcmodel_args
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        seq_tag_size = 3
        self.seq_tag_size = seq_tag_size
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, seq_tag_size,prgcmodel_args.dropout)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, seq_tag_size,prgcmodel_args.dropout)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, seq_tag_size,prgcmodel_args.dropout)
        # global correspondence
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1,prgcmodel_args.dropout)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size,config.num_labels,prgcmodel_args.dropout)
        self.rel_embedding = nn.Embedding(config.num_labels,config.hidden_size)

        self.loss_fn1 = nn.CrossEntropyLoss(reduction='none',ignore_index=-100)
        self.loss_fn2 = nn.BCEWithLogitsLoss(reduction='none')


    def get_model_lr(self):
        return super(TransformerForPRGC, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.sequence_tagging_sub, self.config.task_specific_params['learning_rate_for_task']),
            (self.sequence_tagging_obj, self.config.task_specific_params['learning_rate_for_task']),
            (self.sequence_tagging_sum, self.config.task_specific_params['learning_rate_for_task']),
            (self.global_corres, self.config.task_specific_params['learning_rate_for_task']),
            (self.rel_judgement, self.config.task_specific_params['learning_rate_for_task']),
            (self.rel_embedding, self.config.task_specific_params['learning_rate_for_task']),
        ]

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)


    def predict_seqs_output(self, sequence_outputs, attention_masks, pred_rels):
        device = sequence_outputs.device
        bs,seqlen,h = sequence_outputs.size()
        # (bs, rel_num)
        preds_seqs_list = []
        for sequence_output, attention_mask,pred_rel in zip(sequence_outputs, attention_masks, pred_rels):
            potential_rels = []
            for rel_id,one_rel in enumerate(pred_rel):
                if one_rel > self.prgcmodel_args.rel_threshold:
                    potential_rels.append(rel_id)
            if potential_rels:
                potential_rels = torch.tensor(potential_rels, dtype=torch.long).to(device)
                sequence_output = torch.repeat_interleave(sequence_output.unsqueeze(0), len(potential_rels), 0)
                #(bs / sum(x_i), 2,seq_len, tags)
                seqs_preds = self.forward_sub_obj_output(sequence_output, potential_rels)
                preds_seqs_list.append(seqs_preds)
            else:
                preds_seqs_list.append(torch.zeros(size=(1,2,seqlen,self.seq_tag_size)))

        #b,n,2,s,t
        return preds_seqs_list

    #,b,s,h ; b,
    def forward_sub_obj_output(self, sequence_output, potential_rels):

        seqlen, h = sequence_output.size()[1:]
        #potential_rels: (bs,), only in train stage.
        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        # b,s,h
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seqlen, h)
        if self.prgcmodel_args.emb_fusion == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            sub_output = self.sequence_tagging_sub(decode_input)
            obj_output = self.sequence_tagging_obj(decode_input)
        elif self.prgcmodel_args.emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            sub_output, obj_output = self.sequence_tagging_sum(decode_input)
        else:
            raise ValueError('bad emb_fusion',self.prgcmodel_args.emb_fusion)
        return torch.stack([sub_output, obj_output],dim=1)

    #PRGC将关系抽取分解成三个任务：关系判断、实体抽取和主客体对齐
    def compute_loss(self, *args,**batch) -> tuple:
        """
         Args:
             input_ids: (batch_size, seq_len)
             attention_mask: (batch_size, seq_len)
             rel_tags: (bs, rel_num)
             corres_tags: (bs, seq_len, seq_len)
             potential_rels: (bs,), only in train stage.
             seq_tags: (bs, 2, seq_len)
        """
        # get params for experiments
        corres_tags: torch.Tensor = batch.pop('corres_tags', None)
        rel_tags: torch.Tensor = batch.pop('rel_tags', None)
        potential_rels: torch.Tensor = batch.pop('potential_rels', None)
        seq_tags: torch.Tensor = batch.pop('seq_tags', None)

        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        sequence_outputs = outputs[0]
        if self.model.training:
            sequence_outputs = self.dropout(sequence_outputs)

        bs, seqlen, h = sequence_outputs.size()

        if self.prgcmodel_args.ensure_rel:
            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_outputs, attention_mask)
            # (bs, rel_num)
            pred_rels = self.rel_judgement(h_k_avg)
        else:
            pred_rels = torch.tensor([1] * self.rel_num, device=sequence_outputs.device).repeat(bs).reshape(bs, -1)

        # before fuse relation representation
        if self.prgcmodel_args.ensure_corres:
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_outputs.unsqueeze(2).expand(-1, -1, seqlen, -1)  # (bs, s, s, h)
            obj_extend = sequence_outputs.unsqueeze(1).expand(-1, seqlen, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            pred_corres = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            pred_corres = self.global_corres(pred_corres).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        if self.training:
            # (bs/sum(x_i), 2, s, tag_size)
            pred_seqs = self.forward_sub_obj_output(sequence_outputs, potential_rels)

            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            # loss rel
            if self.prgcmodel_args.ensure_rel:
                loss_rel = self.loss_fn2(pred_rels, rel_tags.float()).mean()

            # loss corres
            if self.prgcmodel_args.ensure_corres:
                corres_mask_tmp = corres_mask.view(bs, -1)
                # corres_tags = corres_tags.view(bs, -1)
                corres_tags = corres_tags.float()
                loss_matrix = (self.loss_fn2(pred_corres.view(bs,-1), corres_tags.view(bs,-1)) * corres_mask_tmp).sum() / corres_mask_tmp.sum()

            # sequence label loss

            mask = attention_mask.view(-1)
            #b,2,s,tags , b, 2, s
            loss_seq1 = self.loss_fn1(pred_seqs[:, 0].reshape(-1, self.seq_tag_size), seq_tags[:, 0].reshape(-1).long())
            loss_seq2 = self.loss_fn1(pred_seqs[:, 1].reshape(-1, self.seq_tag_size), seq_tags[:, 1].reshape(-1).long())
            loss_seq1 = (loss_seq1 * mask).sum() / mask.sum()
            loss_seq2 = (loss_seq2 * mask).sum() / mask.sum()
            loss_seq = loss_seq1 + loss_seq2
            loss_seq = loss_seq.sum()

            loss_dict = {
                'loss': loss_seq + loss_matrix + loss_rel,
                'loss_seq': loss_seq,
                'loss_matrix': loss_matrix,
                'loss_rel': loss_rel,
            }
            # (bs, rel_num)
            if self.prgcmodel_args.ensure_rel:
                pred_rels = torch.sigmoid(pred_rels)

            # (bs, seq_len, seq_len)
            if self.prgcmodel_args.ensure_corres:
                pred_corres = torch.sigmoid(pred_corres) * corres_mask
            else:
                pred_corres = None

            outputs = (loss_dict,pred_rels, pred_seqs, pred_corres)
        else:
            # (bs, rel_num)
            if self.prgcmodel_args.ensure_rel:
                pred_rels = torch.sigmoid(pred_rels)

            #b,n,2,s,tags
            pred_seqs = self.predict_seqs_output(sequence_outputs, attention_mask, pred_rels)
            # (bs, seq_len, seq_len)
            if self.prgcmodel_args.ensure_corres:
                pred_corres = torch.sigmoid(pred_corres) * corres_mask

            else:
                pred_corres = None
            outputs = (pred_rels, pred_seqs, pred_corres)
        #evaluate
        if seq_tags is not None and not self.training:
            outputs = (None,) + outputs
        return outputs
