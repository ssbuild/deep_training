# @Time    : 2022/11/16 22:24
# @Author  : tk
# @FileName: contrast.py

import torch
from torch.nn.functional import cross_entropy


def compute_loss_of_similarity(y_pred: torch.Tensor):
    idxs = torch.arange(0, y_pred.size()[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    labels = torch.eq(idxs_1, idxs_2).to(y_pred.device).float()
    y_pred = torch.norm(y_pred,dim=1)
    similarities = torch.mm(y_pred, torch.transpose(y_pred,dim0=1,dim1=0))  # 相似度矩阵
    similarities = similarities - torch.eye(y_pred.size()[0]).to(y_pred.device) * 1e12  # 排除对角线
    similarities = similarities * 30  # scale
    loss = cross_entropy(similarities, labels)
    return loss

