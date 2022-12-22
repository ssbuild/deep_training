# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 10:13
import argparse
import typing
from typing import Any

import numpy as np
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

from ...data_helper import PrefixModelArguments
from .transformer import TransformerModel
from ..layers.crf import CRF
from ..layers.prefix_encoder import PrefixEncoder
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, f1_metric_for_pointer
from ..losses.loss_globalpointer import loss_for_pointer
from ..metrics.pointer import metric_for_pointer
from ..utils import get_value_from_args

__all__ = [
    'PrefixTransformerForModel',
    'PrefixTransformerForSequenceClassification',
    'PrefixTransformerForTokenClassification',
    'PrefixTransformerForCRF'
]




class PrefixTransformerForModel(TransformerModel):
    def __init__(self, *args: Any, **kwargs: Any):
        prompt_args = get_value_from_args('prompt_args', PrefixModelArguments, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.prompt_args = prompt_args
        config = self.config
        config.pre_seq_len = prompt_args.pre_seq_len
        if prompt_args.prompt_type != 0:
            config.prefix_projection = prompt_args.prefix_projection
            config.prefix_hidden_size = prompt_args.prefix_hidden_size

        self.num_labels = config.num_labels
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        self.embeddings = self.model.embeddings
        if prompt_args.prompt_type != 0:
            self.prefix_encoder = PrefixEncoder(config)
        else:
            self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        the_model_param = 0
        for name, param in self.model.named_parameters():
            the_model_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - the_model_param
        print('total param is {}'.format(total_param))  # 9860105
        #function
        self._get_prompt : typing.Callable = self.get_prompt_0 if prompt_args.prompt_type == 0 else self.get_prompt_1
        self._get_transformer_outputs : typing.Callable = self.get_transformer_outputs_0 if prompt_args.prompt_type == 0 else self.get_transformer_outputs_1


    def get_model_lr(self):
        return super(PrefixTransformerForModel, self).get_model_lr() + [
            (self.prefix_encoder, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def get_prompt_0(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def get_prompt_1(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        # (self.n_layer * 2,batch,n_head,pre_seq_len,n_embd)
        # (24,batch,12,16,64)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, **batch):
        return self._get_transformer_outputs(**batch)

    def get_transformer_outputs_0(self,**batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.shape[0]

        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=batch.get('position_ids',None),
            token_type_ids=batch.get('token_type_ids',None),
        )
        prompts = self._get_prompt(batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            # past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]

        return (sequence_output,first_token_tensor)


    def get_transformer_outputs_1(self,**batch):
        input_ids = batch['input_ids']
        attention_mask = batch.pop('attention_mask')
        batch_size = input_ids.shape[0]
        past_key_values = self._get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        batch['attention_mask'] = attention_mask
        outputs = self.model(
            **batch,
            past_key_values=past_key_values,
        )
        return outputs


class PrefixTransformerForSequenceClassification(PrefixTransformerForModel):
    def __init__(self,  *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.config.num_labels)

    def get_model_lr(self):
        return super(PrefixTransformerForSequenceClassification, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def compute_loss(self, *args,**batch) -> tuple:
        labels = batch.pop('labels',None)
        outputs = self.model(*args,**batch)
        pooled_output = outputs[1]
        if self.model.training:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            outputs = (loss,logits,labels)
        else:
            outputs = (logits,)
        return outputs




class PrefixTransformerForTokenClassification(PrefixTransformerForModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_labels = self.config.num_labels
        self.classifier = torch.nn.Linear(self.config.hidden_size,  self.config.num_labels)

    def get_model_lr(self):
        return super(PrefixTransformerForTokenClassification, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def compute_loss(self, *args,**batch) -> tuple:
        labels = batch.pop('labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        pooled_output = outputs[1]
        if self.model.training:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the losses
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,logits,labels)
        else:
            outputs = (logits,)
        return outputs




class PrefixTransformerPointer(PrefixTransformerForModel):
    def __init__(self, *args, **kwargs):
        with_efficient = kwargs.pop('with_efficient', True)
        super().__init__(*args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64)

    def get_model_lr(self):
        return super(PrefixTransformerPointer, self).get_model_lr() + [
            (self.pointer_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.pointer_layer(logits, batch['attention_mask'])
        if labels is not None:
            loss = loss_for_pointer(labels, logits)
            f1 = f1_metric_for_pointer(labels, logits)
            loss_dict = {'loss': loss, 'f1': f1}
            outputs = (loss_dict, logits, labels)
        else:
            outputs = (logits,)
        return outputs

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        id2label = self.config.id2label
        threshold = 1e-8
        y_preds, y_trues = [], []
        for o in outputs:
            logits, label = o['outputs']
            logits[:, :, [0, -1]] -= np.inf
            logits[:, :, :, [0, -1]] -= np.inf
            assert len(logits) == len(label)
            for p, t in zip(logits, label):
                a_result = []
                for (l, s, e) in zip(*np.where(p > threshold)):
                    a_result.append((l, s, e))
                y_preds.append(a_result)
                b_result = []
                for (l, s, e) in zip(*np.where(t > threshold)):
                    b_result.append((l, s, e))
                y_trues.append(b_result)
        f1, str_report = metric_for_pointer(y_trues, y_preds, id2label)
        print(f1)
        print(str_report)
        self.log('val_f1', f1, prog_bar=True)


class PrefixTransformerForCRF(PrefixTransformerForModel):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels)

    def get_model_lr(self):
        return super(PrefixTransformerForCRF, self).get_model_lr() + [
            (self.classifier, self.config.task_specific_params['learning_rate']),
            (self.crf, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.classifier(logits)
        tags = self.crf.decode(logits, attention_mask)
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (loss, tags, labels)
        else:
            outputs = (tags,)
        return outputs


