# -*- coding: utf-8 -*-
# @Time    : 2023/1/13 14:14
# reference: https://github.com/shuxinyin/SimCSE-Pytorch
from typing import Union, Optional, Callable, Any

import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer

from .transformer import TransformerModel
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss

__all__ = [
    'TransformerForESimcse'
]

class TransformerForESimcse(TransformerModel):
    def __init__(self, *args,**kwargs):
        pooling = kwargs.pop('pooling','cls')
        gamma = kwargs.pop('gamma', 0.95)
        super(TransformerForESimcse, self).__init__(*args,**kwargs)
        self.pooling = pooling
        self.gamma = gamma
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

    def optimizer_step(self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_idx: int = 0,
            optimizer_closure: Optional[Callable[[], Any]] = None,
            on_tpu: bool = False,
            using_lbfgs: bool = False,
            **kwargs
    ) -> None:
        # update params
        optimizer.step(closure=optimizer_closure)
        model = self.model
        momentum_encoder = self.momentum_encoder

        #  Momentum Contrast Encoder Update
        for encoder_param, moco_encoder_param in zip(model.parameters(), momentum_encoder.parameters()):
            # print("--", moco_encoder_param.data.shape, encoder_param.data.shape)
            moco_encoder_param.data = self.gamma \
                                      * moco_encoder_param.data \
                                      + (1. - self.gamma) * encoder_param.data


    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        if self.model.training:
            neg_num = batch.pop('neg_num').cpu().numpy().tolist()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            loss_logits = []
            for i in range(n):
                inputs = {}
                inputs['input_ids'] = input_ids[:, i]
                inputs['attention_mask'] = attention_mask[:, i]
                loss_logits.append(self.forward_for_pos_hidden(**inputs))
            for i in range(neg_num):
                input_ids = batch['input_ids' + str(i)]
                attention_mask = batch['attention_mask' + str(i)]
                inputs = {}
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = attention_mask
                loss_logits.append(self.forward_for_neg_hidden(**inputs))
            loss = self.loss_fn(loss_logits)
            outputs = (loss,)
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            logits1 = self.forward_for_pos_hidden(*args, **batch)
            logits2 = self.forward_for_pos_hidden(*args, **inputs)
            labels = torch.squeeze(labels, dim=-1)
            outputs = (None, logits1,logits2, labels)
        else:
            logits = self.forward_for_pos_hidden(*args, **batch)
            outputs = (logits,)
        return outputs
