# -*- coding: utf-8 -*-
# @Time    : 2023/1/16 15:44

import torch
from torch import nn
from .transformer import TransformerModel
from ..losses.loss_infonce import InfoNCE

__all__ = [
    'TransformerForInfoNce'
]


class TransformerForInfoNce(TransformerModel):
    def __init__(self, *args, **kwargs):
        pooling = kwargs.pop('pooling', 'cls')
        temperature = kwargs.pop('temperature', 0.1)
        vector_size = kwargs.pop('vector_size', 512)
        super(TransformerForInfoNce, self).__init__(*args, **kwargs)
        config = self.config
        self.pooling = pooling
        self.feat_head = nn.Linear(config.hidden_size, vector_size, bias=False)
        self.loss_fn = InfoNCE(temperature=temperature,negative_mode='paired', reduction='sum')

    def get_model_lr(self):
        return super(TransformerForInfoNce, self).get_model_lr() + [
            (self.feat_head, self.config.task_specific_params['learning_rate_for_task'])
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
            simcse_logits = self.feat_head(outputs[1])
            simcse_logits = torch.tanh(simcse_logits)
        else:
            raise ValueError('not support pooling', self.pooling)
        return simcse_logits

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        if self.model.training:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            pos, neg = [], []
            for i in range(n):
                inputs = {}
                inputs['input_ids'] = input_ids[:, i]
                inputs['attention_mask'] = attention_mask[:, i]
                obj = pos if i < 2 else neg
                obj.append(self.forward_for_hidden(**inputs))
            if neg:
                neg_key = torch.stack(neg, dim=1)
            else:
                neg_key = None
            query, pos_key = pos
            loss = self.loss_fn(query, pos_key, neg_key)
            outputs = (loss,)
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            logits1 = self.forward_for_hidden(*args, **batch)
            logits2 = self.forward_for_hidden(*args, **inputs)
            labels = torch.squeeze(labels, dim=-1)
            outputs = (None, logits1, logits2, labels)
        else:
            logits = self.forward_for_hidden(*args, **batch)
            outputs = (logits,)

        return outputs
