# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 16:33

from torch import nn

class RDropLoss(nn.Module):
    def __init__(self,alpha=4, reduction='mean'):
        super(RDropLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self,inputs,targets):
        loss = self.entropy(inputs,targets.long())
        loss_kl = self.kl(inputs[::2],inputs[1::2]) + self.kl(inputs[1::2],inputs[::2])
        return loss + loss_kl / 4 * self.alpha
