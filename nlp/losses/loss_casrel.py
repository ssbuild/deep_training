# @Time    : 2022/12/5 0:32
# @Author  : tk
# @FileName: loss_casrel.py
import torch
from torch import nn


class LossForCasRel(nn.Module):
    def __init__(self, reduction='none'):
        super(LossForCasRel, self).__init__()
        self.reduction = reduction
        self.criterion = nn.BCELoss(reduction=self.reduction)

    def forward(self, inputs, targets,mask=None):
        subject_preds, object_preds = inputs
        subject_labels, object_labels = targets
        if mask is None:
            mask = torch.ones(subject_preds.size()[:2],dtype=torch.float32).to(subject_preds.device)
        loss1 = self.criterion(subject_preds, subject_labels)
        loss1 = loss1.mean(-1)
        loss1 = (loss1 * mask).sum() / mask.sum()

        loss2 = self.criterion(object_preds, object_labels)
        loss2 = loss2.mean(-1).sum(-1)
        loss2 = (loss2 * mask).sum() / mask.sum()
        loss = loss1 + loss2
        return loss