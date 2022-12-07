# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 12:19
import torch
from torch import nn


class TplinkerLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(TplinkerLoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self,inputs: torch.Tensor,targets: torch.Tensor):
        loss = self.criterion(inputs.view(-1, inputs.size()[-1]),targets.view(-1))
        return loss

class TplinkerPlusLoss(nn.Module):
    def __init__(self,inf=1e12):
        super(TplinkerPlusLoss, self).__init__()
        self.inf = inf

    def GHM(self, gradient, bins=10, beta=0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        '''
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid((gradient - avg) / std)  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(gradient_norm, 0, 0.9999999)  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        #         print()
        #         print("hits_vec: {}".format(hits_vec))
        #         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights
        #         print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(gradient_norm.size()[0], gradient_norm.size()[1], 1)
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients



    def forward(self, y_pred, y_true, ghm=False):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        # shaking_seq_len = y_pred.size()[1]
        # y_pred = torch.reshape(torch.transpose(y_pred, 1, 2),(-1,shaking_seq_len))
        # y_true = torch.reshape(torch.transpose(y_true, 1, 2),(-1,shaking_seq_len))

        tag_size = y_pred.size()[-1]
        y_pred = torch.reshape(y_pred,(-1,tag_size))
        y_true = torch.reshape(y_true,(-1,tag_size))

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * self.inf  # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * self.inf  # mask the pred oudtuts of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        if ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        return (neg_loss + pos_loss).mean()
