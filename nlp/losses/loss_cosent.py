# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 16:52
import torch
from torch import nn
from torch.nn import  functional as F



class CoSentLoss(nn.Module):
    def __init__(self):
        super(CoSentLoss, self).__init__()

    def forward(self,y_true, y_pred):
        """排序交叉熵
        y_true：标签/打分，y_pred：句向量
        """
        y_true = y_true[::2, 0]
        y_true = y_true[:, None] < y_true[None, :].float()
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_pred = torch.reshape(y_pred - (1 - y_true) * 1e12, [-1])
        y_pred = torch.cat([[0], y_pred], dim=0)
        return torch.logsumexp(y_pred,dim=0)