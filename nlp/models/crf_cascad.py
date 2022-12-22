# @Time    : 2022/12/9 19:49
# @Author  : tk
# @FileName: crf_cascad.py
'''
如果需要考虑实体类别，那么就需要扩展 BIO 的 tag 列表，给每个“实体类型”都分配一个 B 与 I 的标签，但是当类别数较多时，标签词表规模很大，相当于在每个字上都要做一次类别数巨多的分类任务，不科学，也会影响效果
从这个点出发，就尝试把 NER 改成一个多任务学习的框架，两个任务，一个任务用来单纯抽取实体，一个任务用来判断实体类型，
'''
import copy
import torch
from torch import nn
from .transformer import TransformerModel
from ..layers.crf import CRF
__all__ = [
    'TransformerForCascadCRF'
]


class Chunk:
    l, s, e = -1, -1, -1
    def reset(self):
        self.l = -1
        self.s = -1
        self.e = -1

def get_entities(logits_tags,ents_logits):
    length = len(logits_tags)
    chunks = []
    chunk = Chunk()

    def reset_chunk(chunk: Chunk):
        chunk.reset()

    for indx,(T,L) in enumerate(zip(logits_tags,ents_logits)):
        if T == 'S':
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            chunk.s = indx
            chunk.e = indx
            chunk.l = L
            chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        elif T == 'B':
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
            chunk.s = indx
            chunk.l = L
        elif T == 'I' and chunk.s != -1:
            if L == chunk.l:
                chunk.e = indx
            else:
                reset_chunk(chunk)
            if indx == length - 1:
                if chunk.e != -1:
                    chunks.append(copy.deepcopy(chunk))
                    reset_chunk(chunk)
        elif T == 'O' and chunk.s != -1:
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        else:
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
    if chunk.e != -1:
        chunks.append(copy.deepcopy(chunk))
        reset_chunk(chunk)

    return [(chunk.l,chunk.s - 1,chunk.e - 1) for chunk in chunks]

def extract_lse(outputs,id2seqs):
    batch_result = []
    for crf_logits, ents_logits in zip(outputs[0],outputs[1].argmax(-1)):
        batch_result.append(get_entities([id2seqs[l] for l in crf_logits], ents_logits))
    return batch_result


class TransformerForCascadCRF(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForCascadCRF, self).__init__(*args,**kwargs)
        config = self.config

        seqs2id = self.config.task_specific_params['seqs2id']
        ents2id = self.config.task_specific_params['ents2id']
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seqs_classifier = nn.Linear(config.hidden_size, len(seqs2id))
        self.crf = CRF(num_tags=len(seqs2id))
        self.ents_classifier = nn.Linear(config.hidden_size, len(ents2id))
        self.cross_loss = nn.CrossEntropyLoss(reduction='none')


    def get_model_lr(self):
        return super(TransformerForCascadCRF, self).get_model_lr() + [
            (self.dropout, self.config.task_specific_params['learning_rate_for_task']),
            (self.seqs_classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.ents_classifier, self.config.task_specific_params['learning_rate_for_task']),
            (self.crf, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        seqs_labels: torch.Tensor = batch.pop('seqs_labels',None)
        ents_labels: torch.Tensor = batch.pop('ents_labels',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        seqs_logits = self.seqs_classifier(logits)
        crf_tags = self.crf.decode(seqs_logits, attention_mask)
        ents_logits = self.ents_classifier(logits)
        if seqs_labels is not None:
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