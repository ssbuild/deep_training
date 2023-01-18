# -*- coding: utf-8 -*-
# @Time    : 2023/1/16 15:15
# reference: https://github.com/kongds/Prompt-BERT

import typing
from dataclasses import field, dataclass

import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss

__all__ = [
    'TransformerForPromptbertcse'
]


@dataclass
class PromptBertcseArguments:
    mask_embedding_sentence: typing.Optional[bool] = field(
        default=True,
        metadata={
        }
    )
    mask_embedding_sentence_template: typing.Optional[str] = field(
        default='*cls*_This_sentence_:_\'*sent_0*\'_means*mask*.*sep+*',
        metadata={
        }
    )
    mask_embedding_sentence_bs: typing.Optional[str] = field(
        default='This sentence of "',
        metadata={
        }
    )
    mask_embedding_sentence_es: typing.Optional[str] = field(
        default='" means [MASK].',
        metadata={
        }
    )
    mask_embedding_sentence_different_template: typing.Optional[str] = field(
        default='',
        metadata={
        }
    )
    mask_embedding_sentence_bs2: typing.Optional[str] = field(
        default='This sentence of "',
        metadata={
        }
    )
    mask_embedding_sentence_es2: typing.Optional[str] = field(
        default='" means [MASK].',
        metadata={
        }
    )
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

    mask_embedding_sentence_autoprompt_freeze_prompt: typing.Optional[bool] = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_random_init: typing.Optional[bool] = field(
        default=False,
        metadata={
        }
    )


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
    def __init__(self, *args, **kwargs):
        promptbertcse_args: PromptBertcseArguments = kwargs.pop('promptbertcse_args')
        tokenizer = kwargs.pop('tokenizer')
        super(TransformerForPromptbertcse, self).__init__(*args, **kwargs)
        self.promptbertcse_args = promptbertcse_args
        config = self.config
        self.mlp = MLPLayer(config)
        self.loss_fn = MultipleNegativesRankingLoss()
        self.config.mask_token_id = self.config.task_specific_params['mask_token_id']

        bs = tokenizer.encode(promptbertcse_args.mask_embedding_sentence_bs, add_special_tokens=False)
        es = tokenizer.encode(promptbertcse_args.mask_embedding_sentence_es, add_special_tokens=False)
        if len(promptbertcse_args.mask_embedding_sentence_different_template) > 0:
            bs2 = tokenizer.encode(promptbertcse_args.mask_embedding_sentence_bs2, add_special_tokens=False)
            es2 = tokenizer.encode(promptbertcse_args.mask_embedding_sentence_es2, add_special_tokens=False)
        else:
            bs2, es2 = bs, es
        mask_embedding_template = tokenizer.encode(
            promptbertcse_args.mask_embedding_sentence_bs + promptbertcse_args.mask_embedding_sentence_es)

        print('template bs', bs)
        print('template es', es)
        print('template mask_embedding_template', tokenizer.decode(mask_embedding_template))
        print('template mask_embedding_template', mask_embedding_template)

        assert len(mask_embedding_template) == len(bs) + len(es) + 2
        assert mask_embedding_template[1:-1] == bs + es

        if len(promptbertcse_args.mask_embedding_sentence_different_template) > 0:
            mask_embedding_template2 = tokenizer.encode(promptbertcse_args.mask_embedding_sentence_bs2 + \
                                                        promptbertcse_args.mask_embedding_sentence_es2)
            print('d template mask_embedding_template', tokenizer.decode(mask_embedding_template2))
            print('d template mask_embedding_template', mask_embedding_template2)
        else:
            mask_embedding_template2 = None

        self.model_extra = {
            'bs': bs,
            'es': es,
            'bs2': bs2,
            'es2': es2,
            'mask_embedding_template': mask_embedding_template,
            'mask_embedding_template2': mask_embedding_template2
        }
        model = self.model
        if self.promptbertcse_args.mask_embedding_sentence_autoprompt:
            # register p_mbv in init, avoid not saving weight
            self.p_mbv = torch.nn.Parameter(torch.zeros(10))
            for param in self.model.parameters():
                param.requires_grad = False

            mask_index = self.model_extra['mask_embedding_template'].index(self.config.mask_token_id)
            index_mbv = self.model_extra['mask_embedding_template'][1:mask_index] + self.model_extra['mask_embedding_template'][ mask_index + 1:-1]

            self.dict_mbv = index_mbv
            self.fl_mbv = [i <= 3 for i, k in enumerate(index_mbv)]
            p_mbv_w = model.embeddings.word_embeddings.weight[self.dict_mbv].clone()
            self.register_parameter(name='p_mbv', param=torch.nn.Parameter(p_mbv_w))
            if self.promptbertcse_args.mask_embedding_sentence_autoprompt_freeze_prompt:
                self.p_mbv.requires_grad = False

            if self.promptbertcse_args.mask_embedding_sentence_autoprompt_random_init:
                self.p_mbv.data.normal_(mean=0.0, std=0.02)
        else:
            self.dict_mbv = None
            self.fl_mbv = None

    def get_model_lr(self):
        current = [(self.mlp, self.config.task_specific_params['learning_rate_for_task']),
                   (getattr(self, 'p_mbv', None), self.config.task_specific_params['learning_rate_for_task']),
                   (getattr(self, 'dict_mbv', None), self.config.task_specific_params['learning_rate_for_task']),
                   (getattr(self, 'fl_mbv', None), self.config.task_specific_params['learning_rate_for_task']), ]
        return super(TransformerForPromptbertcse, self).get_model_lr() + [
            item for item in current if item[0] is not None
        ]

    def get_delta(self, template_token, device, length=50):
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
                    # print(d_inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
                    # print(d_inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
                    d_inputs_embeds[b, index] = self.p_mbv[i]
            else:
                d_inputs_embeds = None
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(length, 1).long()
            if not self.promptbertcse_args.mask_embedding_sentence_delta_no_position:
                d_position_ids[:, len(self.model_extra['bs']) + 1:] += torch.arange(length).to(device).unsqueeze(-1)

            template_len = d_input_ids.shape[1]
            inputs = dict(input_ids=d_input_ids if d_inputs_embeds is None else None,
                          inputs_embeds=d_inputs_embeds,
                          position_ids=d_position_ids)
            delta = self.forward_for_hidden(**inputs)
            return delta, template_len

    def forward_for_hidden(self, *args, **batch):
        input_ids = batch['input_ids']
        outputs = self.model(*args, **batch, output_hidden_states=True, )
        logits = outputs[0][input_ids == self.config.mask_token_id]
        if self.promptbertcse_args.mask_embedding_sentence_org_mlp:
            logits = self.mlp(logits)
        return logits

    def forward_for_test(self, *args, **batch):
        input_ids = batch['input_ids']

        if self.promptbertcse_args.mask_embedding_sentence_delta:
            batch_size = input_ids.size(0)
            N = max(int(batch_size * 1.5), 128)
            device = input_ids.device
            d_input_ids = torch.Tensor([self.model_extra['mask_embedding_template']]).repeat(N, 1).to(device).long()
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(N, 1).long()
            if not self.promptbertcse_args.mask_embedding_sentence_delta_no_position:
                d_position_ids[:, len(self.model_extra['bs']) + 1:] += torch.arange(N).to(device).unsqueeze(-1)
            inputs = dict(input_ids=d_input_ids, position_ids=d_position_ids)
            delta = self.forward_for_hidden(**inputs)
            delta.requires_grad = False
            template_len = d_input_ids.shape[1]

        # if self.promptbertcse_args.mask_embedding_sentence:
        #     new_input_ids = []
        #     bs = torch.LongTensor(self.model_extra['bs']).to(input_ids.device)
        #     es = torch.LongTensor(self.model_extra['es']).to(input_ids.device)
        #
        #     for i in input_ids:
        #         ss = i.shape[0]
        #         d = i.device
        #         ii = i[i != self.config.pad_token_id]
        #         ni = [ii[:1], bs]
        #         if ii.shape[0] > 2:
        #             ni += [ii[1:-1]]
        #         ni += [es, ii[-1:]]
        #         if ii.shape[0] < i.shape[0]:
        #             ni += [i[i == self.config.pad_token_id]]
        #         ni = torch.cat(ni)
        #         try:
        #             assert ss + bs.shape[0] + es.shape[0] == ni.shape[0]
        #         except:
        #             print(ss + bs.shape[0] + es.shape[0])
        #             print(ni.shape[0])
        #             print(i.tolist())
        #             print(ni.tolist())
        #             assert 0
        #         new_input_ids.append(ni)
        #     input_ids = torch.stack(new_input_ids, dim=0)
        #     attention_mask = (input_ids != self.config.pad_token_id).long()
        #
        #     batch.clear()
        #     batch['input_ids'] = input_ids
        #     batch['attention_mask'] = attention_mask

        if self.promptbertcse_args.mask_embedding_sentence_autoprompt:
            inputs_embeds = self.model.embeddings.word_embeddings(input_ids)
            with torch.no_grad():
                p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
                b = torch.arange(input_ids.shape[0]).to(input_ids.device)
                for i, k in enumerate(self.dict_mbv):
                    if self.fl_mbv[i]:
                        index = ((input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((input_ids == k) * -p).min(-1)[1]
                    inputs_embeds[b, index] = self.p_mbv[i]

            batch['input_ids'] = None
            batch['inputs_embeds'] = inputs_embeds

        pooler_output = self.forward_for_hidden(**batch)
        if self.promptbertcse_args.mask_embedding_sentence:
            if self.promptbertcse_args.mask_embedding_sentence_delta:
                blen = batch['attention_mask'].sum(-1) - template_len
                pooler_output -= delta[blen]
        return pooler_output

    def compute_loss(self, *args, **batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        if self.training:
            input_ids = batch['input_ids']
            batch_size, n,seqlen = input_ids.size()
            device = input_ids.device
            if self.promptbertcse_args.mask_embedding_sentence_delta:
                N = max(int(batch_size * n * 1.5),128)
                delta, template_len = self.get_delta([self.model_extra['mask_embedding_template']], device,
                                                     length=N)
                if len(self.promptbertcse_args.mask_embedding_sentence_different_template) > 0:
                    delta1, template_len1 = self.get_delta([self.model_extra['mask_embedding_template2']], device,
                                                           length=N)
            attention_mask = batch['attention_mask']
            logits_list = []
            for i in range(n):
                inputs = {
                    'input_ids': input_ids[:, i],
                    'attention_mask': attention_mask[:, i]
                }
                logits_list.append(self.forward_for_hidden(*args, **inputs))
            pooler_output = torch.cat(logits_list, dim=0)

            if self.promptbertcse_args.mask_embedding_sentence_delta:
                if len(self.promptbertcse_args.mask_embedding_sentence_different_template) > 0:
                    pooler_output = pooler_output.view(batch_size, n, -1)
                    attention_mask = attention_mask.view(batch_size, n, -1)
                    blen = attention_mask.sum(-1) - template_len
                    torch.clamp_min(blen, 0, out=blen)
                    pooler_output[:, 0, :] -= delta[blen[:, 0]]
                    blen = attention_mask.sum(-1) - template_len1
                    torch.clamp_min(blen, 0, out=blen)
                    pooler_output[:, 1, :] -= delta1[blen[:, 1]]
                    if n == 3:
                        pooler_output[:, 2, :] -= delta1[blen[:, 2]]
                else:
                    blen = attention_mask.view(-1, attention_mask.size(2)).sum(-1) - template_len
                    torch.clamp_min(blen, 0, out=blen)
                    pooler_output -= delta[blen]
            logits_list = torch.split(pooler_output, batch_size, dim=0)
            loss = self.loss_fn(logits_list)
            outputs = (loss,)
        elif labels is not None:
            batch2 = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    batch2[k.replace('2', '')] = batch.pop(k)
            logits = self.forward_for_test(*args, **batch)
            logits2 = self.forward_for_test(*args, **batch2)
            labels = torch.squeeze(labels, dim=-1)
            outputs = (None, logits, logits2, labels)
        else:
            logits = self.forward_for_test(*args, **batch)
            outputs = (logits,)
        return outputs