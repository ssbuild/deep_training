# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 10:13
import argparse
from typing import Any

import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from .transformer import TransformerModel
from ..layers.prefix_encoder import PrefixEncoder



class PrefixTransformerForModel(TransformerModel):
    def __init__(self, config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config, train_args, *args, **kwargs)

        config.pre_seq_len = train_args.pre_seq_len
        config.prefix_projection = train_args.prefix_projection
        config.prefix_hidden_size = train_args.prefix_hidden_size

        self.num_labels = config.num_labels

        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.model.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        attrs = [model]
        if hasattr(self,'classifier'):
            attrs += [self.classifier]
        opt = []
        for a in attrs:
            opt += [
                {
                    "params": [p for n, p in a.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay, "lr": self.hparams.learning_rate,
                },
                {
                    "params": [p for n, p in a.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": self.hparams.learning_rate,
                },
            ]
        optimizer = AdamW(opt, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def get_prompt(self, batch_size):
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


    def get_transformer_outputs(self,batch):
        input_ids = batch['input_ids']
        attention_mask = batch.pop('attention_mask')
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        batch['attention_mask'] = attention_mask
        outputs = self(
            **batch,
            past_key_values=past_key_values,
        )
        return outputs



class PrefixTransformerForSequenceClassification(PrefixTransformerForModel):
    def __init__(self, config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config, train_args, *args, **kwargs)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)


    def get_loss_and_logits(self,batch):
        labels = batch.pop('labels')
        outputs = self.get_transformer_outputs(batch)
        pooled_output = outputs[1]
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
        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output
        #
        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return loss,logits

    def training_step(self, batch, batch_idx):
        loss ,_ = self.get_loss_and_logits(batch)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["labels"]
        val_loss, logits = self.get_loss_and_logits(batch)
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        _ ,logits = self.get_loss_and_logits(x)
        return logits


class PrefixTransformerForTokenClassification(PrefixTransformerForModel):
    def __init__(self, config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config, train_args, *args, **kwargs)
        self.num_labels = config.num_labels
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)


    def get_loss_and_outputs(self, batch):
        labels = batch.pop('labels')
        attention_mask = batch['attention_mask']
        outputs = self.get_transformer_outputs(batch)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss,logits

    def training_step(self, batch, batch_idx):
        loss ,_ = self.get_loss_and_outputs(batch)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["labels"]
        val_loss, logits = self.get_loss_and_outputs(batch)
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        _ ,logits = self.get_loss_and_outputs(batch)
        return logits