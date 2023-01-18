# @Time    : 2022/12/10 0:44
# @Author  : tk
# @FileName: w2ner.py
'''
reference: https://github.com/ljynlp/W2NER
'''
import typing
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from deep_training.nlp.metrics.pointer import metric_for_pointer
from deep_training.nlp.models.transformer import TransformerModel
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..layers.norm import LayerNorm
from ..layers.seq_pointer import seq_masking
from ..layers.w2ner import CoPredictor,ConvolutionLayer

@dataclass
class W2nerArguments:
    use_last_4_layers: typing.Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )
    dist_emb_size: typing.Optional[int] = field(
        default=20,
        metadata={
            "help": (
                ""
            )
        },
    )
    type_emb_size: typing.Optional[int] = field(
        default=20,
        metadata={
            "help": (
                ""
            )
        },
    )
    lstm_hid_size: typing.Optional[int] = field(
        default=768,
        metadata={
            "help": (
                ""
            )
        },
    )

    conv_hid_size: typing.Optional[int] = field(
        default=96,
        metadata={
            "help": (
                ""
            )
        },
    )
    biaffine_size: typing.Optional[int] = field(
        default=768,
        metadata={
            "help": (
                ""
            )
        },
    )
    ffnn_hid_size: typing.Optional[int] = field(
        default=128,
        metadata={
            "help": (
                ""
            )
        },
    )
    dilation: typing.Optional[tuple] = field(
        default=(1,2,3),
        metadata={
            "help": (
                ""
            )
        },
    )

    emb_dropout: typing.Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                ""
            )
        },
    )
    conv_dropout: typing.Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                ""
            )
        },
    )
    out_dropout: typing.Optional[float] = field(
        default=0.3333,
        metadata={
            "help": (
                ""
            )
        },
    )




def extract_lse(outputs):
    batch_result = []
    for logits,seqlen in zip(outputs[0],outputs[1]):
        # l,l,n
        logits = logits.argmax(-1)
        logits = logits[:seqlen, :seqlen]
        logits[[0, -1]] = 0
        logits[:, [0, -1]] = 0
        logits_pred = np.tril(logits)
        lse = []
        for e,s in zip(*np.where(logits_pred > 1)):
            l = logits_pred[e,s]
            lse.append((l-2,s-1,e-1))
        batch_result.append(lse)
    return batch_result


class TransformerForW2ner(TransformerModel):
    def __init__(self,w2nerArguments: W2nerArguments, *args,**kwargs):
        super(TransformerForW2ner, self).__init__(*args,**kwargs)

        self.w2nerArguments = w2nerArguments
        config = self.config

        self.lstm_hid_size = w2nerArguments.lstm_hid_size
        self.conv_hid_size = w2nerArguments.conv_hid_size

        lstm_input_size = 0
        lstm_input_size += config.hidden_size

        self.dropout = nn.Dropout(w2nerArguments.emb_dropout)

        self.dis_embs = nn.Embedding(20, w2nerArguments.dist_emb_size)
        self.reg_embs = nn.Embedding(3, w2nerArguments.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, w2nerArguments.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = w2nerArguments.lstm_hid_size + w2nerArguments.dist_emb_size + w2nerArguments.type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, w2nerArguments.conv_hid_size, w2nerArguments.dilation, w2nerArguments.conv_dropout)


        #first two label is used
        self.predictor = CoPredictor(config.num_labels + 2,
                                     w2nerArguments.lstm_hid_size, 
                                     w2nerArguments.biaffine_size,
                                     w2nerArguments.conv_hid_size * len(w2nerArguments.dilation), 
                                     w2nerArguments.ffnn_hid_size,
                                     w2nerArguments.out_dropout)

        self.cln = LayerNorm(w2nerArguments.lstm_hid_size, w2nerArguments.lstm_hid_size)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')


    def get_model_lr(self):
        return super(TransformerForW2ner, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.dis_embs, self.config.task_specific_params['learning_rate_for_task']),
            (self.reg_embs, self.config.task_specific_params['learning_rate_for_task']),
            (self.encoder, self.config.task_specific_params['learning_rate_for_task']),
            (self.convLayer, self.config.task_specific_params['learning_rate_for_task']),
            (self.predictor, self.config.task_specific_params['learning_rate_for_task']),
            (self.criterion, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def mlp(self, bert_embs, grid_mask2d, dist_inputs, pieces2word,attr_mask):
        '''
        # :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return: blln
        '''

        sent_length = attr_mask.sum(-1)
        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        if self.training:
            word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln([word_reps.unsqueeze(2), word_reps])

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        outputs = seq_masking(outputs,attr_mask,1,value=1e-12)
        outputs = seq_masking(outputs,attr_mask,2,value=1e-12)
        return outputs,sent_length

    def compute_loss(self, *args,**batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        grid_mask2d= batch.pop('grid_mask2d',None)
        dist_inputs= batch.pop('dist_inputs',None)
        pieces2word= batch.pop('pieces2word',None)
        attention_mask = batch['attention_mask']

        if self.w2nerArguments.use_last_4_layers:
            outputs = self(**batch,output_hidden_states=True)
            logits = torch.stack(outputs[2][-4:], dim=-1).mean(-1)
        else:
            outputs = self.model(*args,**batch)
            logits = outputs[0]

        grid_mask2d = grid_mask2d.clone()
        logits,seqlens = self.mlp(logits, grid_mask2d, dist_inputs, pieces2word,attention_mask)


        if labels is not None:
            labels = labels[grid_mask2d].long()
            loss = self.criterion(logits[grid_mask2d],labels)
            outputs = (loss,logits,seqlens,labels)
        else:
            outputs = (logits,seqlens,)
        return outputs

    # def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
    #     label2id = self.config.label2id
    #     y_preds, y_trues = [], []
    #     eval_labels = self.eval_labels
    #     for i, o in enumerate(outputs):
    #         logits,seqlens, _ = o['outputs']
    #         y_preds.extend(extract_lse([logits,seqlens]))
    #         bs = len(logits)
    #         y_trues.extend(eval_labels[i * bs: (i + 1) * bs])
    #
    #     f1, str_report = metric_for_pointer(y_trues, y_preds, label2id)
    #     print(f1)
    #     print(str_report)
    #     self.log('val_f1', f1, prog_bar=True)