# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 14:43
import torch
from torch import nn


class BCELossForSplinker(nn.Module):
    def __init__(self, ):
        super(BCELossForSplinker, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(torch.sum(loss,dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss