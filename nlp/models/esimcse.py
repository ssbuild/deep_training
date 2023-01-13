# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 14:14
# 参考实现: https://github.com/shuxinyin/SimCSE-Pytorch

import torch
from .transformer import TransformerModel
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss

__all__ = [
    'TransformerForESimcse'
]

class TransformerForESimcse(TransformerModel):
    def __init__(self, *args,**kwargs):
        pooling = kwargs.pop('pooling','cls')
        super(TransformerForESimcse, self).__init__(*args,**kwargs)
        self.pooling = pooling
        # config = self.config
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.momentum_encoder = TransformerModel(*args,**kwargs)
        self.loss_fn = MultipleNegativesRankingLoss()

    def get_model_lr(self):
        return super(TransformerForESimcse, self).get_model_lr()

    def pooling_output(self,outputs):
        if self.pooling == 'cls':
            simcse_logits = outputs[0][:, 0]
        elif self.pooling == 'pooler':
            simcse_logits = outputs[1]
        elif self.pooling == 'last-avg':
            last = outputs[0].transpose(1, 2)  # [batch, 768, seqlen]
            simcse_logits = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = outputs[2][1].transpose(1, 2)  # [batch, 768, seqlen]
            last = outputs[2][-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            simcse_logits = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        # elif self.pooling == 'reduce':
        #     simcse_logits = self.sim_head(outputs[1])
        #     simcse_logits = torch.tanh(simcse_logits)
        else:
            raise ValueError('not support pooling', self.pooling)
        return simcse_logits

    def forward_for_pos_hidden(self, *args, **batch):
        outputs = self.model(*args, **batch, output_hidden_states=True, )
        return self.pooling_output(outputs)

    def forward_for_neg_hidden(self, *args, **batch):
        outputs = self.momentum_encoder(*args, **batch, output_hidden_states=True, )
        return self.pooling_output(outputs)


    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        if self.model.training:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            pos, neg = [], []
            for i in range(n):
                inputs = {}
                inputs['input_ids'] = input_ids[:, i]
                inputs['attention_mask'] = attention_mask[:, i]
                pos.append(self.forward_for_pos_hidden(**inputs))

            loss_logits_list = [*pos]
            if 'input_ids2' in batch:
                input_ids = batch['input_ids2']
                attention_mask = batch['attention_mask2']
                n = input_ids.size(1)
                for i in range(n):
                    inputs = {}
                    inputs['input_ids'] = input_ids[:, i]
                    inputs['attention_mask'] = attention_mask[:, i]
                    neg.append(self.forward_for_neg_hidden(**inputs))
                neg_key = torch.stack(neg, dim=1)
                loss_logits_list.append(neg_key)
            loss = self.loss_fn(loss_logits_list)
            outputs = (loss,)
        elif labels is not None:
            logits = self.forward_for_pos_hidden(*args, **batch)
            outputs = (None, logits, labels)
        else:
            logits = self.forward_for_pos_hidden(*args, **batch)
            outputs = (logits,)
        return outputs
