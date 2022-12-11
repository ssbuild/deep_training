# @Time    : 2022/12/11 13:57
# @Author  : tk
# @FileName: loss_mutiheadlinker.py

import torch
from deep_training.nlp.losses.loss_globalpointer import multilabel_categorical_crossentropy
from torch import nn

class MutiheadlinkerLoss(nn.Module):
    def __init__(self,reduction = 'none'):
        super(MutiheadlinkerLoss, self).__init__()
        self.reduction = reduction
        self.entropy_fn = nn.BCELoss(reduction=self.reduction)

    def forward(self,inputs: torch.Tensor,targets: torch.Tensor,mask: torch.Tensor):
        mask = mask.float()
        tags = inputs.size()[2]
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, tags,-1)  # batch x seq x rel x seq
        selection_loss = self.entropy_fn(inputs,targets.float())
        selection_loss = selection_loss.masked_select(selection_mask.bool()).sum(-1).sum(-1).sum(-1)
        selection_loss /= (mask.sum(-1).sum(-1).sum(-1) + 1e-12)
        return selection_loss


class MutiheadlinkerLossEx(nn.Module):
    def __init__(self):
        super(MutiheadlinkerLossEx, self).__init__()
    def forward(self,inputs: torch.Tensor,targets: torch.Tensor):
        selection_loss = multilabel_categorical_crossentropy(targets,inputs)
        return selection_loss