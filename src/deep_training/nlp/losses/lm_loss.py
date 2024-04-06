# -*- coding: utf-8 -*-
# @Time:  21:08
# @Author:XIE392
# @File：lm_loss.py

from torch import nn, Tensor


class LM_loss(nn.CrossEntropyLoss):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,input: Tensor, target: Tensor,with_shift=True):
        target = target.long()
        if with_shift:
            input = input[..., :-1, :].contiguous()
            target = target[..., 1:].contiguous()
        return super().forward(input.view(-1, input.size(-1)),target.view(-1))