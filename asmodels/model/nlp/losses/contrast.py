# @Time    : 2022/11/16 22:24
# @Author  : tk
# @FileName: contrast.py

import torch
from torch.nn.functional import cross_entropy


def compute_simcse_loss(y_pred: torch.Tensor):
    batch_size = y_pred.size()[0]
    idxs = torch.arange(0, batch_size )
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    labels = torch.eq(idxs_1, idxs_2).to(y_pred.device).float()
    y_pred = torch.layer_norm(y_pred,normalized_shape=y_pred.shape)
    similarities = torch.mm(y_pred, torch.transpose(y_pred,1,0))
    similarities = similarities - torch.eye(batch_size).to(y_pred.device) * 1e12
    similarities = similarities * 30  # scale
    loss = cross_entropy(similarities, labels)
    return loss

