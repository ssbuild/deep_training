# @Time    : 2022/11/10 21:54
# @Author  : tk
# @FileName: seqpointer.py
import torch
import torch.nn as nn

__all__ = [
    'f1_metric_for_pointer',
    'PointerLayer',
    'EfficientPointerLayer'
]




def f1_metric_for_pointer(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0).float()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)


def seq_masking(x:torch.Tensor,mask,axis,value=-1e12):
    if mask is None:
        return x
    for _ in range(axis - 1):
        mask = torch.unsqueeze(mask,1)
    for _ in range(len(x.size()) -len(mask.size())):
        mask = torch.unsqueeze(mask, -1)
    x = x * mask + (1 - mask) * value
    return x

class PointerLayer(nn.Module):
    def __init__(self, in_hidden_size, heads, head_size, RoPE=True, tril_mask=True, inf=1e12):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.hidden_size = in_hidden_size
        self.dense = nn.Linear(self.hidden_size, self.heads * self.head_size * 2)

        self.inf = inf
        self.RoPE = RoPE
        self.tril_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        #embeddings = position_ids * indices
        #(1,seq_len) * (output_dim // 2)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings



    def forward(self, context_output, mask = None):


        self.device = context_output.device
        batch_size,seq_len = context_output.size()[0:2]


        # outputs:(batch_size, seq_len, heads*head_size*2)
        outputs = self.dense(context_output)

        # outputs: heads 个 (batch_size, seq_len,2 * head_size)
        outputs = torch.split(outputs, self.head_size * 2, dim=-1)


        # outputs:(batch_size, heads,heads,head_size *2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, heads, head_size)
        qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]  # TODO:修改为Linear获取？



        if self.RoPE:
            # pos_emb:(batch_size, seq_len, head_size)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)


        logits = seq_masking(logits, mask, 2, -self.inf)
        logits = seq_masking(logits, mask, 3, -self.inf)

        if self.tril_mask:
            #排除下三角
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * self.inf

        return logits / self.head_size ** 0.5



class EfficientPointerLayer(nn.Module):
    def __init__(self, in_hidden_size, heads, head_size, RoPE=True, tril_mask=True, inf=1e12):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.hidden_size = in_hidden_size
        self.dense1 = nn.Linear(self.hidden_size, self.head_size * 2)
        self.dense2 = nn.Linear(self.hidden_size, self.heads * 2)

        self.inf = inf
        self.RoPE = RoPE
        self.tril_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        #embeddings = position_ids * indices
        #(1,seq_len) * (output_dim // 2)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings


    def forward(self, context_output, mask):
        self.device = context_output.device

        batch_size,seq_len = context_output.size()[0:2]
        # outputs:(batch_size, seq_len, head_size*2)
        outputs = self.dense1(context_output)
        # qw,kw:(batch_size, seq_len, head_size)
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, head_size)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size)
            #cos_pos,sin_pos: (batch_size, seq_len, 1, head_size)
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

            # cos_pos = torch.repeat_interleave(pos[..., 1::2], 2, -1)
            # sin_pos = torch.repeat_interleave(pos[..., ::2], 2, -1)
            # qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            # qw2 = torch.reshape(qw2, qw.shape)
            # qw = qw * cos_pos + qw2 * sin_pos
            # kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            # kw2 = torch.reshape(kw2, kw.shape)
            # kw = kw * cos_pos + kw2 * sin_pos

        #(batch,seq,seq)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5
        #(batch,heads * 2,seq)
        outputs_dense = self.dense2(context_output)
        bias = torch.einsum('bnh->bhn',outputs_dense ) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        logits = seq_masking(logits, mask, 2, -self.inf)
        logits = seq_masking(logits, mask, 3, -self.inf)

        if self.tril_mask:
            #排除下三角
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * self.inf
        return logits