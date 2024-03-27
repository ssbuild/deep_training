# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:06
import torch


def build_masks_and_position_ids_glm(batch_input_ids, ctxlens, max_len = None):
    if max_len is None:
        max_len = batch_input_ids.size(1)

    batch_position_ids, batch_attention_mask = [], []
    for input_ids, context_length in zip(batch_input_ids, ctxlens):
        if context_length.dim() == 1:
            context_length = context_length.squeeze(dim=-1)

        mask_position = context_length - 1
        position_ids = list(range(context_length)) + [mask_position] * (max_len - context_length)
        block_position_ids = [0] * context_length + list(range(1, max_len - context_length + 1))

        attention_mask = torch.ones((1, max_len, max_len))
        attention_mask = torch.tril(attention_mask)
        attention_mask[..., :context_length] = 1
        attention_mask = (attention_mask < 0.5)

        batch_position_ids.append(torch.stack((torch.tensor(position_ids), torch.tensor(block_position_ids))))
        batch_attention_mask.append(attention_mask)

    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_position_ids = torch.stack(batch_position_ids, dim=0)
    return batch_attention_mask,batch_position_ids
