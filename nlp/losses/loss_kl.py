# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 16:09

from torch import nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self,reduction='none',log_target: bool = False):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.loss_fn = nn.KLDivLoss(reduction=self.reduction,log_target=log_target)

    def forward(self,inputs,pad_mask = None):
        p,q = inputs
        p_loss = self.loss_fn(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
        q_loss = self.loss_fn(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)



        loss = (p_loss + q_loss) / 2
        return loss

