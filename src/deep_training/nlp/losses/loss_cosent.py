# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 16:52
import torch
from torch import nn
from torch.nn import  functional as F


def cat_even_odd_reorder(logits1,logits2):
    mid_logits = torch.cat([logits1, logits2], dim=0)
    index = torch.arange(0, mid_logits.size(0)).to(logits1.device)
    index = torch.cat([index[::2], index[1::2]])
    index = torch.unsqueeze(index, 1)
    if index.size() != mid_logits.size():
        index = index.expand(*mid_logits.size())
    mid_logits_state = torch.zeros_like(mid_logits).to(logits1.device)
    mid_logits_state = torch.scatter(mid_logits_state, dim=0, index=index, src=mid_logits)
    return mid_logits_state

class CoSentLoss(nn.Module):
    def __init__(self):
        super(CoSentLoss, self).__init__()

    def forward(self,y_true, y_pred):
        """排序交叉熵
        y_true：标签/打分，y_pred：句向量
        """
        y_true = y_true[::2, 0]
        y_true = y_true[:, None] < y_true[None, :]
        y_true = y_true.float()
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_pred = y_pred.float()
        y_pred = torch.reshape(y_pred - (1 - y_true) * 1e12, [-1])
        zero = torch.tensor([0],dtype=torch.float32,requires_grad=True).to(y_pred.device)
        y_pred = torch.cat([zero, y_pred], dim=0)
        return torch.logsumexp(y_pred,dim=0)



