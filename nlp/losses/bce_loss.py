# @Time    : 2022/11/30 21:36
# @Author  : tk
# @FileName: bce_loss.py
import torch
from torch import nn


class BCELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, logits, labels, mask=None):
        loss = self.criterion(logits, labels)
        if mask is not None:
            mask = mask.float()
            loss = loss * mask.unsqueeze(-1)
            loss = torch.sum(torch.sum(loss,dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss

