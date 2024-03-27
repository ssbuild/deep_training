# @Time    : 2022/11/16 22:24
# @Author  : tk
# @FileName: contrast.py

import torch
from torch import nn
from torch.nn import functional as F


class SimcseLoss(nn.Module):
    def __init__(self,scale = 20,reduction='sum',ignore_index=-100):
        super(SimcseLoss, self).__init__()
        self.scale = scale
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self,inputs):
        batch_size = inputs.size(0)
        idxs = torch.arange(0, batch_size)
        idxs = torch.reshape(idxs, (-1, 2))
        idxs = torch.cat([idxs[:, 1:], idxs[:, :1]], dim=1)
        y_true = torch.flatten(idxs).to(inputs.device).long()
        # sim = torch.mm(y_pred, torch.transpose(y_pred,1,0))
        sim = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(batch_size).to(inputs.device) * 1e12
        sim = sim * self.scale  # scale
        loss = F.cross_entropy(sim, y_true, reduction='sum')
        return loss
