# -*- coding: utf-8 -*-
# @Time    : 2023/1/16 15:15
import typing
from dataclasses import field, dataclass

import torch
from torch import nn
from .transformer import TransformerModel
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
__all__ = [
    'TransformerForPromptbertcse'
]


@dataclass
class PromptBertcseArguments:
    mask_embedding_sentence_org_mlp: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )
    mask_embedding_sentence_delta_freeze: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
    })
    mask_embedding_sentence_autoprompt: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        })
    mask_embedding_sentence_delta_no_position: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        })

    mask_embedding_sentence_delta: typing.Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                ""
            )
        })

    mask_embedding_sentence_different_template: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        })



class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class TransformerForPromptbertcse(TransformerModel):
    def __init__(self, *args,**kwargs):
        promptbertcse_args : PromptBertcseArguments = kwargs.pop('promptbertcse_args')
        super(TransformerForPromptbertcse, self).__init__(*args,**kwargs)
        self.promptbertcse_args = promptbertcse_args
        config = self.config
        self.mlp = MLPLayer(config)
        self.loss_fn = MultipleNegativesRankingLoss()
        self.config.mask_token_id = self.config.task_specific_params['mask_token_id']

    
        self.dict_mbv = None
        self.fl_mbv = None
        self.p_mbv = None
        self.bs = None
        self.es = None
        self.mask_embedding_template = None
        self.mask_embedding_template2 = None

    def get_model_lr(self):
        return super(TransformerForPromptbertcse, self).get_model_lr() + [
            (self.mlp, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def get_delta(self,template_token,device, length=50):
        with torch.set_grad_enabled(not self.promptbertcse_args.mask_embedding_sentence_delta_freeze):
            d_input_ids = torch.Tensor(template_token).repeat(length, 1).to(device).long()
            if self.promptbertcse_args.mask_embedding_sentence_autoprompt:
                d_inputs_embeds = self.model.embeddings.word_embeddings(d_input_ids)
                p = torch.arange(d_input_ids.shape[1]).to(d_input_ids.device).view(1, -1)
                b = torch.arange(d_input_ids.shape[0]).to(d_input_ids.device)
                for i, k in enumerate(self.dict_mbv):
                    if self.fl_mbv[i]:
                        index = ((d_input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((d_input_ids == k) * -p).min(-1)[1]
                    #print(d_inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
                    #print(d_inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
                    d_inputs_embeds[b, index] = self.p_mbv[i]
            else:
                d_inputs_embeds = None
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(length, 1).long()
            if not self.promptbertcse_args.mask_embedding_sentence_delta_no_position:
                d_position_ids[:, len(self.bs)+1:] += torch.arange(length).to(device).unsqueeze(-1)

            template_len = d_input_ids.shape[1]
            inputs = dict(input_ids=d_input_ids if d_inputs_embeds is None else None ,
                              inputs_embeds=d_inputs_embeds,
                              position_ids=d_position_ids,  output_hidden_states=True, return_dict=True)

            delta = self.forward_for_hidden(**inputs)
            return delta, template_len

    def forward_for_hidden(self, *args, **batch):
        input_ids = batch['input_ids']
        outputs = self.model(*args, **batch, output_hidden_states=True, )
        logits = outputs[0][input_ids == self.config.mask_token_id]
        logits = torch.squeeze(logits,dim=1)
        if self.promptbertcse_args.mask_embedding_sentence_org_mlp:
            logits = self.mlp(logits)
        return logits

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        if self.training:

            input_ids = batch['input_ids']
            device = input_ids.device

            if self.promptbertcse_args.mask_embedding_sentence_delta:
                delta, template_len = self.get_delta([self.mask_embedding_template],device)
                if len(self.promptbertcse_args.mask_embedding_sentence_different_template) > 0:
                    delta1, template_len1 = self.get_delta([self.mask_embedding_template2],device)


            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            loss_logits = []
            for i in range(n):
                inputs = {
                    'input_ids': input_ids[:, i],
                    'attention_mask': attention_mask[:, i]
                }
                loss_logits.append(self.forward_for_hidden(*args, **inputs))

            pooler_output = torch.cat(loss_logits, dim=0)
            batch_size, num_sent = pooler_output.size()[:2]
            if self.promptbertcse_args.mask_embedding_sentence_delta:
                if len(self.promptbertcse_args.mask_embedding_sentence_different_template) > 0:
                    pooler_output = pooler_output.view(batch_size, num_sent, -1)
                    attention_mask = attention_mask.view(batch_size, num_sent, -1)
                    blen = attention_mask.sum(-1) - template_len
                    pooler_output[:, 0, :] -= delta[blen[:, 0]]
                    blen = attention_mask.sum(-1) - template_len1
                    pooler_output[:, 1, :] -= delta1[blen[:, 1]]
                    if num_sent == 3:
                        pooler_output[:, 2, :] -= delta1[blen[:, 2]]
                else:
                    blen = attention_mask.sum(-1) - template_len
                    pooler_output -= delta[blen]

            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            if cls.model_args.mask_embedding_sentence_delta and cls.model_args.mask_embedding_sentence_org_mlp:
                # ignore the delta and org
                pass
            else:
                pooler_output = cls.mlp(pooler_output)


            loss = self.loss_fn(loss_logits)
            outputs = (loss,)
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            simcse_logits = self.forward_for_hidden(*args, **batch)
            simcse_logits2 = self.forward_for_hidden(*args, **inputs)
            labels = torch.squeeze(labels, dim=-1)
            outputs = (None, simcse_logits, simcse_logits2, labels)
        else:
            simcse_logits = self.forward_for_hidden(*args, **batch)
            outputs = (simcse_logits,)
        return outputs
