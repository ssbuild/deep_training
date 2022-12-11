# @Time    : 2022/12/11 15:52
# @Author  : tk
# @FileName: mhslayer.py
import math
import torch
from torch import nn
from .seq_pointer import seq_masking

class MhsLayer(nn.Module):
    def __init__(self,in_features,out_features,inf=1e-12):
        super(MhsLayer, self).__init__()
        self.inf = inf
        self.in_features = in_features
        self.out_features = out_features
        self.selection_u = nn.Linear(in_features, in_features)
        self.selection_v = nn.Linear(in_features, in_features)
        self.selection_uv = nn.Parameter(torch.empty((2 * in_features, out_features)))

        bound = 1 / math.sqrt(self.selection_uv.size(1))
        nn.init.uniform_(self.selection_uv, -bound, bound)

    def forward(self,inputs,mask):
        B, L = inputs.size()[:2]
        u = self.selection_u(inputs).unsqueeze(1).expand(B, L, L, -1)
        v = self.selection_v(inputs).unsqueeze(2).expand(B, L, L, -1)
        uv = torch.cat([u, v], dim=-1)
        #logits_select = torch.einsum('bijh,hr->birj',uv ,self.selection_uv)
        logits_select = torch.einsum('bijh,hr->brij', uv, self.selection_uv)
        logits_select = seq_masking(logits_select, mask, 2, -self.inf)
        logits_select = seq_masking(logits_select, mask, 3, -self.inf)
        return logits_select
