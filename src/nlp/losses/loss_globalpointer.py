# -*- coding: utf-8 -*-
# @Time    : 2022/11/24 8:57
import torch

def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_for_pointer(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss



def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False,epsilon=1e-7,inf=1e12):
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + inf
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

    y_pos_2 = torch.gather(y_pred,-1, y_true)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred,-1, y_true)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), epsilon, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss

def loss_for_gplinker(y_true: torch.Tensor, y_pred: torch.Tensor,mask_zero=True):
    shape = y_pred.shape
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = torch.reshape(y_pred, (shape[0], -1, torch.prod(torch.tensor(shape[2:])).numpy()))
    loss = sparse_multilabel_categorical_crossentropy(y_true.long(),y_pred,mask_zero)
    return loss.sum(dim=1).mean()