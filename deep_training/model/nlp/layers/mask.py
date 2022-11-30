# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 8:42

import torch

__all__ = [
    'lm_mask',
    'unilm_mask'
]


def lm_mask(s: torch.Tensor):
    seq_len = s.size()[1]
    idxs = torch.arange(0, seq_len)
    mask = idxs[None, :] <= idxs[:, None]
    return mask.long()

def unilm_mask(s: torch.Tensor):
    idxs = torch.cumsum(s, dim=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    return mask.long()

