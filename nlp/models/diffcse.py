# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 12:26
# reference: https://github.com/voidism/DiffCSE , https://arxiv.org/pdf/2204.10298.pdf
import typing
from dataclasses import dataclass, field

import torch
from torch import nn
from .transformer import TransformerModel
from transformers.activations import ACT2FN
from transformers import AutoModelForMaskedLM
from ..losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss

__all__ = [
    'TransformerForDiffcse'
]


@dataclass
class DiffcselArguments:
    pooling: typing.Optional[str] = field(
        default='cls',
        metadata={
            "help": (
                "one of [cls , reduce]"
            )
        },
    )

    vector_size: typing.Optional[int] = field(
        default=512,
        metadata={
            "help": (
                ""
            )
        },
    )


    batchnorm: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use two-layer mlp for the pooler"
            )
        },
    )

    lambda_weight:typing.Optional[float] = field(
        default=0.05,
        metadata={
            "help": (
                ""
            )
        },
    )

    mlm_probability: typing.Optional[float] = field(
        default=0.15,
        metadata={
            "help": (
                ""
            )
        },
    )

    encoder_with_mlp: typing.Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                ""
            )
        },
    )

    discriminator_with_mlp: typing.Optional[bool] = field(
        default = True,
        metadata={
            "help": (
                ""
            )
        },
    )


    num_encoder_layer: typing.Optional[int] = field(
        default=6,
        metadata={
            "help": (
                ""
            )
        },
    )

    num_discriminator_layer: typing.Optional[int] = field(
        default=6,
        metadata={
            "help": (
                ""
            )
        },
    )


    generator_model_type: typing.Optional[str] = field(
        default='bert',
        metadata={
            "help": (
                ""
            )
        },
    )
    generator_model_name_or_path: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )

    generator_config_name: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": (
                ""
            )
        },
    )


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

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

class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class TransformerForDiffcse(TransformerModel):
    def __init__(self, *args,**kwargs):
        diffcse_args: DiffcselArguments = kwargs.pop('diffcse_args', None)
        generator_config = kwargs.pop('generator_config',None)
        super(TransformerForDiffcse, self).__init__(*args,**kwargs)
        self.generator_config = generator_config
        self.diffcse_args = diffcse_args
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if generator_config is not None:
            self.discriminator_cls = TransformerModel(*args, **kwargs)
            self.discriminator = self.discriminator_cls.model
            # self.generator = DistilBertForMaskedLM.from_pretrained(
            #     'distilbert-base-uncased') if cls.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(
            #     cls.model_args.generator_name)
            self.discriminator.electra_head = torch.nn.Linear(768, 2)
            self.discriminator_embeddings_fn = self.discriminator.embeddings.forward
            self.discriminator.mlp = MLPLayer(config) if not self.diffcse_args.batchnorm else ProjectionMLP(config)


            model_name_or_path = diffcse_args.generator_model_name_or_path
            if model_name_or_path is not None:
                self.generator = AutoModelForMaskedLM.from_pretrained(model_name_or_path,
                                                                    from_tf=bool(".ckpt" in model_name_or_path),
                                                                    config=self.generator_config)
            else:
                self.generator = AutoModelForMaskedLM.from_pretrained(self.generator_config)

        else:
            self.classifier2, self.decoder, self.loss_fn = None, None, None

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.generator_config.pad_token_id,reduction='none')
        self.loss_fn_cse = MultipleNegativesRankingLoss()


    def get_model_lr(self):
        return super(TransformerForDiffcse, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.discriminator, self.config.task_specific_params['learning_rate_for_task']),
        ]


    def pooling_output(self, outputs,num_layers):
        pooling = self.diffcse_args.pooling
        if pooling == 'cls':
            simcse_logits = outputs[2][num_layers][:, 0]
        elif pooling== 'pooler':
            simcse_logits = outputs[1]
        elif pooling == 'last-avg':
            last = outputs[2][num_layers] # [batch, 768, seqlen]
            simcse_logits = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif pooling == 'first-last-avg':
            first = outputs[2][1].transpose(1, 2)  # [batch, 768, seqlen]
            last = outputs[2][num_layers].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            simcse_logits = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        # elif pooling == 'reduce':
        #     simcse_logits = self.sim_head(outputs[1])
        #     simcse_logits = torch.tanh(simcse_logits)
        else:
            raise ValueError('not support pooling', self.pooling)
        return simcse_logits

    def forward_for_hidden(self, *args, **batch):
        outputs = self.model(*args, **batch, output_hidden_states=True, )
        return self.pooling_output(outputs, self.diffcse_args.num_encoder_layer)

    def forward_generator_output(self, *args, **batch):
        with torch.no_grad():
            outputs = self.generator(**batch,output_hidden_states=True,)
            preds = outputs[0].argmax(-1)
        preds[:,0] = batch['input_ids'][0][0]
        return preds

    def forward_discriminator_output(self, *args, **batch):
        cls_input = batch.pop('cls_input')
        def forward_fn(*args,**kwargs):
            embedding_output = self.discriminator_embeddings_fn(*args,**kwargs)
            embedding_output = torch.cat([cls_input.unsqueeze(1), embedding_output[:, 1:, :]], dim=1)
            return embedding_output
        setattr(self.discriminator.embeddings,'forward',forward_fn)
        outputs = self.discriminator(*args, **batch, output_hidden_states=True, )
        outputs = outputs[2][self.diffcse_args.num_discriminator_layer]
        if self.diffcse_args.discriminator_with_mlp:
            outputs = self.discriminator.mlp(outputs)
        return outputs

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        #训练
        if self.training:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            n = input_ids.size(1)
            loss_logits = []
            for i in range(n):
                inputs = {}
                inputs['input_ids'] = input_ids[:, i]
                inputs['attention_mask'] = attention_mask[:, i]
                loss_logits.append(self.forward_for_hidden(**inputs))

            loss_cse = self.loss_fn_cse(loss_logits)

            input_ids2 = batch['mlm_input_ids']
            n = input_ids2.size(1)
            mlm,mlm_labels,attention_masks = [],[],[]
            for i in range(n):
                inputs = {}
                inputs['input_ids'] = input_ids2[:, i] * attention_mask[:, i]
                inputs['attention_mask'] = attention_mask[:, i]
                g_pred = self.forward_generator_output(**inputs)
                replaced_label = (g_pred != input_ids[:, i]) * inputs['attention_mask']
                inputs['input_ids'] = g_pred * inputs['attention_mask']
                #b,s,h
                cls_input = loss_logits[i]
                cls_input.view(-1,cls_input.size(-1))
                inputs['cls_input'] = cls_input
                mlm.append(self.forward_discriminator_output(**inputs))
                mlm_labels.append(replaced_label)
                attention_masks.append(inputs['attention_mask'])

            mlm = torch.cat(mlm,dim=0)
            mlm_labels = torch.cat(mlm_labels, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0).view(-1)
            prediction_scores = self.discriminator.electra_head(mlm)
            mlm_loss = torch.sum(self.loss_fn(prediction_scores.view(-1, 2), mlm_labels.long().view(-1)) * attention_masks) / torch.sum(attention_masks)
            loss = loss_cse + mlm_loss * self.diffcse_args.lambda_weight
            loss_dict = {
                'loss_cse': loss_cse,
                'mlm_loss': mlm_loss,
                'loss': loss
            }
            outputs = (loss_dict,)
        # 评估
        elif labels is not None:
            inputs = {}
            for k in list(batch.keys()):
                if k.endswith('2'):
                    inputs[k.replace('2', '')] = batch.pop(k)
            labels = torch.squeeze(labels,dim=-1)
            logits1 = self.forward_for_hidden(*args, **batch)
            logits2 = self.forward_for_hidden(*args, **inputs)
            outputs = (None, logits1, logits2, labels)
        else:
            logits = self.forward_for_hidden(*args, **batch)
            outputs = (logits,)
        return outputs

