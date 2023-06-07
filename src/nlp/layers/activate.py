# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 13:47
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x