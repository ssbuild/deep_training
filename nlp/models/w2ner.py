# @Time    : 2022/12/10 0:44
# @Author  : tk
# @FileName: w2ner.py
'''
参考: https://github.com/ljynlp/W2NER
'''
import typing
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from deep_training.nlp.models.transformer import TransformerModel
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..layers.norm import LayerNorm
from ..layers.w2ner import MLP,CoPredictor,ConvolutionLayer,Biaffine

@dataclass
class W2nerArguments:
    use_bert_last_4_layers: typing.Optional[bool] = field(
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




def extract_lse(outputs,id2seqs):...
    # batch_result = []
    # for crf_logits, ents_logits in zip(outputs[0],outputs[1].argmax(-1)):
    #     batch_result.append(get_entities([id2seqs[l] for l in crf_logits], ents_logits))
    # return batch_result


class TransformerForW2ner(TransformerModel):
    def __init__(self,w2nerArguments: W2nerArguments, *args,**kwargs):
        super(TransformerForW2ner, self).__init__(*args,**kwargs)
       
        config = self.config
        self.cross_loss = nn.CrossEntropyLoss(reduction='none')
        self.lstm_hid_size = w2nerArguments.lstm_hid_size
        self.conv_hid_size = w2nerArguments.conv_hid_size

        lstm_input_size = 0
        lstm_input_size += config.hidden_size

        self.dis_embs = nn.Embedding(20, w2nerArguments.dist_emb_size)
        self.reg_embs = nn.Embedding(3, w2nerArguments.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, w2nerArguments.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = w2nerArguments.lstm_hid_size + w2nerArguments.dist_emb_size + w2nerArguments.type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, w2nerArguments.conv_hid_size, w2nerArguments.dilation, w2nerArguments.conv_dropout)
        self.dropout = nn.Dropout(w2nerArguments.emb_dropout)
        self.predictor = CoPredictor(config.num_labels, 
                                     w2nerArguments.lstm_hid_size, 
                                     w2nerArguments.biaffine_size,
                                     w2nerArguments.conv_hid_size * len(w2nerArguments.dilation), 
                                     w2nerArguments.ffnn_hid_size,
                                     w2nerArguments.out_dropout)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size)


    def get_model_lr(self):
        return super(TransformerForW2ner, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.seqs_classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.ents_classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.crf, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def mlp(self, bert_embs, grid_mask2d, dist_inputs, pieces2word,attr_mask):
        '''
        # :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        # bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        # if self.use_bert_last_4_layers:
        #     bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        # else:
        #     bert_embs = bert_embs[0]

        sent_length = attr_mask.sum(-1)
        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        return outputs

    def compute_loss(self,batch,batch_idx):
        labels: torch.Tensor = batch.pop('labels',None)

        if labels is not None:
            grid_mask2d= batch.pop('grid_mask2d',None)
            dist_inputs= batch.pop('dist_inputs',None)
            pieces2word= batch.pop('pieces2word',None)
            sent_length= batch.pop('sent_length',None)
        attention_mask = batch['attention_mask']
        outputs = self(**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits = self.mlp(logits)
        if labels is not None:
            seqs_labels = torch.where(seqs_labels >= 0, seqs_labels, torch.zeros_like(seqs_labels))
            loss1 = self.crf(emissions=seqs_logits, tags=seqs_labels, mask=attention_mask)
            loss2 = self.cross_loss(ents_logits.view(-1,ents_logits.shape[-1]),ents_labels.view(-1))
            attention_mask = attention_mask.float().view(-1)
            loss2 = (loss2 * attention_mask).sum() / (attention_mask.sum() + 1e-12)
            loss_dict = {
                'crf_loss': loss1,
                'ents_loss': loss2,
                'loss': loss1+ loss2
            }
            outputs = (loss_dict,crf_tags,ents_logits,seqs_labels,ents_labels)
        else:
            outputs = (crf_tags,ents_logits)
        return outputs

