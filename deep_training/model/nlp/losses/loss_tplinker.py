# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 12:19
import torch
from torch import nn


class TplinkerLoss(nn.Module):
    def __init__(self,*args,**kwargs):
        super(TplinkerLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self,inputs: torch.Tensor,targets: torch.Tensor):
        loss = self.criterion(inputs.view(-1, inputs.size()[-1]),targets.view(-1))
        return loss
