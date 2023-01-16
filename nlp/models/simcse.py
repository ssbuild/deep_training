# -*- coding: utf-8 -*-
# @Time    : 2023/1/16 15:15
# 1、原论文batch_size=512，这里是batch_size=64（实在跑不起这么壕的batch_size）；
# 2、原论文的学习率是5e-5，这里是1e-5；
# 3、原论文的最优dropout比例是0.1，这里是0.3；
# 4、原论文的无监督SimCSE是在额外数据上训练的，这里直接随机选了1万条任务数据训练；
# 5、原文无监督训练的时候还带了个MLM任务，这里只有SimCSE训练。

import torch
from torch import nn
from .transformer import TransformerModel
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
__all__ = [
    'TransformerForSimcse'
]




class TransformerForSimcse(TransformerModel):
    def __init__(self, *args,**kwargs):
        pooling = kwargs.pop('pooling')
        super(TransformerForSimcse, self).__init__(*args,**kwargs)
        self.pooling = pooling
        config = self.config
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fn = MultipleNegativesRankingLoss()

    def get_model_lr(self):
        return super(TransformerForSimcse, self).get_model_lr() + [
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def forward_for_hidden(self, *args, **batch):
        outputs = self.model(*args, **batch, output_hidden_states=True, )
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
        elif self.pooling == 'reduce':
            simcse_logits = self.sim_head(outputs[1])
            simcse_logits = torch.tanh(simcse_logits)
        else:
            raise ValueError('not support pooling', self.pooling)
        return simcse_logits

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        if self.training:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            loss_logits = []
            for i in range(n):
                inputs = {
                    'input_ids': input_ids[:, i],
                    'attention_mask': attention_mask[:, i]
                }
                loss_logits.append(self.forward_for_hidden(*args, **inputs))
            loss = self.loss_fn(loss_logits)
            outputs = (loss,)
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            simcse_logits = self.forward_for_hidden(*args, **batch)
            simcse_logits2 = self.forward_for_hidden(*args, **inputs)
            labels = torch.squeeze(labels, dim=-1)
            outputs = (None, simcse_logits, simcse_logits2, labels)
        else:
            simcse_logits = self.forward_for_hidden(*args, **batch)
            outputs = (simcse_logits,)
        return outputs
