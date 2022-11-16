# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 10:13
import argparse
from typing import Any
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from .transformer import TransformerModel
from ..layers.prefix_encoder import PrefixEncoder
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, loss_fn, f1_metric
from ..layers.crf import CRF
from ..utils import configure_optimizers

__all__ = [
    'PrefixTransformerForModel',
    'PrefixTransformerForSequenceClassification',
    'PrefixTransformerForTokenClassification',
    'PrefixTransformerForCRF'
]

class PrefixTransformerForModel(TransformerModel):
    def __init__(self,prompt_type, config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config, train_args, *args, **kwargs)

        config.pre_seq_len = train_args.pre_seq_len
        if prompt_type != 0:
            config.prefix_projection = train_args.prefix_projection
            config.prefix_hidden_size = train_args.prefix_hidden_size

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
        if prompt_type != 0:
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
        self.get_prompt = self.get_prompt_0 if prompt_type == 0 else self.get_prompt_1
        self.get_transformer_outputs = self.get_transformer_outputs_0 if prompt_type == 0 else self.get_transformer_outputs_1

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



    def get_transformer_outputs_0(self,batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.shape[0]

        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=batch.get('position_ids',None),
            token_type_ids=batch.get('token_type_ids',None),
        )
        prompts = self.get_prompt(batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self(
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


    def get_transformer_outputs_1(self,batch):
        input_ids = batch['input_ids']
        attention_mask = batch.pop('attention_mask')
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        batch['attention_mask'] = attention_mask
        outputs = self(
            **batch,
            past_key_values=past_key_values,
        )
        return outputs


class PrefixTransformerForSequenceClassification(PrefixTransformerForModel):
    def __init__(self, prompt_type,config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(prompt_type,config, train_args, *args, **kwargs)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def configure_optimizers(self):
        attrs = [(self.model, self.config.task_specific_params['learning_rate']),
                 (self.classifier, self.config.task_specific_params['learning_rate_for_task'])]
        return configure_optimizers(attrs, self.hparams, self.trainer.estimated_stepping_batches)

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
    def __init__(self, prompt_type,config, train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(prompt_type,config, train_args, *args, **kwargs)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def configure_optimizers(self):
        attrs = [(self.model, self.config.task_specific_params['learning_rate']),
                 (self.classifier, self.config.task_specific_params['learning_rate_for_task'])]
        return configure_optimizers(attrs, self.hparams, self.trainer.estimated_stepping_batches)

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
        _ ,logits = self.get_loss_and_outputs(x)
        return logits


class PrefixTransformerPointer(PrefixTransformerForModel):
    def __init__(self,prompt_type, config, train_args: argparse.Namespace,with_efficient=True, *args, **kwargs):
        super().__init__(prompt_type,config, train_args,*args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.pointer_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64)

    def configure_optimizers(self):
        attrs = [(self.model, self.config.task_specific_params['learning_rate']),
                 (self.pointer_layer, self.config.task_specific_params['learning_rate_for_task'])]
        return configure_optimizers(attrs, self.hparams, self.trainer.estimated_stepping_batches)


    def get_loss_and_logits(self,batch):
        labels = batch.pop('labels')
        attention_mask = batch['attention_mask']
        outputs = self.get_transformer_outputs(batch)
        logits = outputs[0]
        logits = self.pointer_layer(logits, attention_mask)

        loss = None
        if labels is not None:
            loss = loss_fn(labels, logits)
        return loss,logits

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        loss ,logits = self.get_loss_and_logits(batch)
        f1 = f1_metric(labels, logits)
        self.log_dict({'train_loss': loss, 'f1': f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch['labels']
        val_loss, logits = self.get_loss_and_logits(batch)
        f1 = f1_metric(labels, logits)
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        _, logits = self.get_loss_and_outputs(x)
        return logits


class PrefixTransformerForCRF(PrefixTransformerForModel):
    def __init__(self,prompt_type, config, train_args: argparse.Namespace,*args, **kwargs):
        super().__init__(prompt_type,config, train_args, *args, **kwargs)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels)

    def configure_optimizers(self):
        attrs =[(self.model, self.config.task_specific_params['learning_rate']),
                 (self.classifier, self.config.task_specific_params['learning_rate']),
                 (self.crf, self.config.task_specific_params['learning_rate_for_task']), ]
        return configure_optimizers(attrs, self.hparams, self.trainer.estimated_stepping_batches)



    def get_loss_and_logits(self,batch):
        labels = batch.pop('labels')
        attention_mask = batch['attention_mask']
        outputs = self.get_transformer_outputs(batch)
        logits = outputs[0]
        logits = self.classifier(logits)
        loss = None
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            # outputs = (-1 * loss,) + outputs
        # else:
        #     # tags = self.crf.decode(logits, attention_mask)
        #     # outputs = (tags,)

        self.log_dict({'train_loss': loss}, prog_bar=True)
        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss,logits

    def training_step(self, batch, batch_idx):
        loss ,logits = self.get_loss_and_logits(batch)
        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch['labels']
        val_loss, logits = self.get_loss_and_logits(batch)
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        _, logits = self.get_loss_and_outputs(x)
        return logits