# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 13:41
#参考实现: https://github.com/hy-struggle/PRGC

import typing
from collections import Counter
from dataclasses import dataclass, field

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




def extract_spoes(outputs: typing.List):
    batch_result = []
    for logits in outputs:
        ...
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
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        seq_tag_size = 3
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, seq_tag_size,prgcmodel_args.dropout)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, seq_tag_size,prgcmodel_args.dropout)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, seq_tag_size,prgcmodel_args.dropout)
        # global correspondence
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1,prgcmodel_args.dropout)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size,config.num_labels,prgcmodel_args.dropout)
        self.rel_embedding = nn.Embedding(config.num_labels,config.hidden_size)

        self.loss_fn1 = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn2 = nn.BCEWithLogitsLoss(reduction='mean')


    def get_model_lr(self):
        return super(TransformerForPRGC, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate']),
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
        rel_tags: torch.Tensor = batch.pop('rel_tags', None)
        potential_rels: torch.Tensor = batch.pop('potential_rels', None)
        seq_tags: torch.Tensor = batch.pop('seq_tags', None)
        corres_tags: torch.Tensor = batch.pop('corres_tags', None)


        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        sequence_output = outputs[0]
        if self.model.training:
            sequence_output = self.dropout(sequence_output)

        device = sequence_output.device
        bs, seqlen, h = sequence_output.size()

        if self.prgcmodel_args.ensure_rel:
            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

            # before fuse relation representation
        if self.prgcmodel_args.ensure_corres:
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seqlen, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seqlen, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None

        if seq_tags is None:
            if self.prgcmodel_args.ensure_rel:
                # (bs, rel_num)
                rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > self.prgcmodel_args.rel_threshold,
                                              torch.ones(rel_pred.size(), device=device),
                                              torch.zeros(rel_pred.size(), device=device))

                # if potential relation is null
                for idx, sample in enumerate(rel_pred_onehot):
                    if 1 not in sample:
                        # (rel_num,)
                        max_index = torch.argmax(rel_pred[idx])
                        sample[max_index] = 1
                        rel_pred_onehot[idx] = sample

                # 2*(sum(x_i),)
                bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot)
                # get x_i
                xi_dict = Counter(bs_idxs.tolist())
                xi = [xi_dict[idx] for idx in range(bs)]

                pos_seq_output = []
                pos_potential_rel = []
                pos_attention_mask = []
                for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                    # (seq_len, h)
                    pos_seq_output.append(sequence_output[bs_idx])
                    pos_attention_mask.append(attention_mask[bs_idx])
                    pos_potential_rel.append(rel_idx)
                # (sum(x_i), seq_len, h)
                sequence_output = torch.stack(pos_seq_output, dim=0)
                # (sum(x_i), seq_len)
                attention_mask = torch.stack(pos_attention_mask, dim=0)
                # (sum(x_i),)
                potential_rels = torch.stack(pos_potential_rel, dim=0)
            # ablation of relation judgement
            else:
                # construct test data
                sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seqlen, h)
                attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seqlen)
                potential_rels = torch.arange(0, self.rel_num, device=device).repeat(bs)

        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        #b,s,h
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seqlen, h)
        if self.prgcmodel_args.emb_fusion == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif self.prgcmodel_args.emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)

        # train
        if seq_tags is not None:
            # calculate loss
            attention_mask = attention_mask.view(-1)
            # sequence label loss
            loss_seq_sub = (self.loss_fn1(output_sub.view(-1, self.seq_tag_size),seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (self.loss_fn1(output_obj.view(-1, self.seq_tag_size),seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            # init
            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            if self.prgcmodel_args.ensure_corres:
                corres_pred = corres_pred.view(bs, -1)
                corres_mask = corres_mask.view(bs, -1)
                corres_tags = corres_tags.view(bs, -1)
                loss_matrix = (self.loss_fn2(corres_pred,corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            if self.prgcmodel_args.ensure_rel:
                loss_rel = self.loss_fn2(rel_pred, rel_tags.float())


            loss_dict = {
                'loss':  loss_seq + loss_matrix + loss_rel,
                'loss_seq': loss_seq,
                'loss_matrix': loss_matrix,
                'loss_rel': loss_rel,
            }

            outputs = (loss_dict, pred_seqs, xi, pred_rels)
        # inference
        else:
            # (sum(x_i), seq_len)
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            if self.prgcmodel_args.ensure_corres:
                corres_pred = torch.sigmoid(corres_pred) * corres_mask
                # (bs, seq_len, seq_len)
                pred_corres_onehot = torch.where(corres_pred > self.prgcmodel_args.corres_threshold,
                                                 torch.ones(corres_pred.size(), device=corres_pred.device),
                                                 torch.zeros(corres_pred.size(), device=corres_pred.device))
                outputs= (pred_seqs, pred_corres_onehot, xi, pred_rels)
            else:
                outputs = (pred_seqs, xi, pred_rels)
        return outputs
