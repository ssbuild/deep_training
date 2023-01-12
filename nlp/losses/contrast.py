# @Time    : 2022/11/16 22:24
# @Author  : tk
# @FileName: contrast.py

import torch
from torch.nn.functional import cross_entropy


def compute_simcse_loss(y_pred: torch.Tensor,scale = 20):
    batch_size = y_pred.size()[0]
    idxs = torch.arange(0, batch_size)
    idxs = torch.reshape(idxs, (-1, 2))
    idxs = torch.cat([idxs[:, 1:], idxs[:, :1]], dim=1)
    labels = torch.reshape(idxs, (-1,)).to(y_pred.device).long()
    similarities = torch.mm(y_pred, torch.transpose(y_pred,1,0))
    similarities = similarities - torch.eye(batch_size).to(y_pred.device) * 1e12
    similarities = similarities * scale  # scale
    loss = cross_entropy(similarities, labels)
    return loss

