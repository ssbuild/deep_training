# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 11:32
import torch
from torch import nn
from ..layers.norm import LayerNorm2 as LayerNorm


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type = None, inner_enc_type= None):
        super().__init__()
        if shaking_type is None:
            shaking_type = 'cln_plus'
        if inner_enc_type is None:
            inner_enc_type = 'mix_pooling'

        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              bidirectional=False,
                                              batch_first=True)
        elif inner_enc_type == "linear":
            self.inner_context_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                # LayerNorm(hidden_size)
            )


    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        elif inner_enc_type == 'linear':
            inner_context = self.inner_context_layer(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens,mask):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        bs, seqlen, hidden_size = seq_hiddens.size()
        mask = (1- mask) * -1000.
        mask = mask.unsqueeze(2).expand(-1,-1,hidden_size)
        seq_hiddens += mask
        shaking_hiddens_list = []
        for ind in range(seqlen):
            repeat_hiddens = seq_hiddens[:, [ind], :].repeat(1, seqlen - ind, 1)
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])
            else:
                raise ValueError('Invalid shaking_type {}'.format(self.shaking_type))
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens