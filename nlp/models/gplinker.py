# @Time    : 2022/11/11 20:15
# @Author  : tk
# @FileName: gp_linker.py
import typing
from itertools import groupby

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer
from ..losses.loss_globalpointer import loss_for_gplinker

__all__ = [
    'TransformerForGplinker'
]
def extract_spoes(outputs: typing.List, threshold=1e-8):
    batch_spoes = []
    for logit1,logit2,logit3 in zip(outputs[0],outputs[1],outputs[2]):
        subjects, objects = set(), set()
        logit1[:, [0, -1]] -= np.inf
        logit1[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(logit1 > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(logit2[:, sh, oh] > threshold)[0]
                p2s = np.where(logit3[:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((sh-1, st-1, p, oh-1, ot-1))
        batch_spoes.append(list(spoes))
    return batch_spoes




class TransformerForGplinker(TransformerModel):
    def __init__(self,  *args, **kwargs):
        with_efficient = kwargs.pop('with_efficient',True)
        super(TransformerForGplinker, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64, RoPE=False,
                                              tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64, RoPE=False,
                                              tril_mask=False)

    def get_model_lr(self):
        return super(TransformerForGplinker, self).get_model_lr() + [
            (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits1 = self.entities_layer(logits, attention_mask)
        logits2 = self.heads_layer(logits, attention_mask)
        logits3 = self.tails_layer(logits, attention_mask)
        if entity_labels is not None:
            loss1 = loss_for_gplinker(entity_labels, logits1)
            loss2 = loss_for_gplinker(head_labels, logits2)
            loss3 = loss_for_gplinker(tail_labels, logits3)
            loss = (loss1 + loss2 + loss3) / 3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_head': loss2,
                         'loss_tail': loss3}
            outputs = (loss_dict, logits1, logits2, logits3,
                       entity_labels, head_labels, tail_labels)
        else:
            outputs = (logits1, logits2, logits3)
        return outputs





class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）
    """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))

def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_,_,  h1, t1) in enumerate(argus):
        for i2, (_,_,  h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def evaluate_events(y_trues: typing.List,y_preds: typing.List,id2label: typing.Dict):
    """评估函数，计算f1、precision、recall
    """

    trues_all,preds_all = [], []

    for one in y_trues:
        events = []
        for evt in one:
            event = []
            for l,s,e in evt:
                event.append((id2label[l],s,e))
            events.append(event)
        trues_all.append(events)

    for one in y_preds:
        events = []
        for evt in one:
            event = []
            for l, s, e in evt:
                event.append((id2label[l], s, e))
            events.append(event)
        preds_all.append(events)


    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别
    for true_events,pred_events in tqdm(zip(trues_all,preds_all),total=len(trues_all)):

        # 事件级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            if any([argu[0].find(u'触发词') != -1 for argu in event]):
                R.append(list(sorted(event)))
        for event in true_events:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1
        ey += len(R)
        ez += len(T)
        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[0].find(u'触发词') == -1:
                    R.append(argu)
        for event in true_events:
            for argu in event:
                if argu[0].find(u'触发词') == -1:
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1
        ay += len(R)
        az += len(T)
    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az
    return e_f1, e_pr, e_rc, a_f1, a_pr, a_rc

def extract_events(outputs,label2id,id2label: dict,threshold: float=1e-8,trigger=True):
    batch_result = []
    for entities,heads,tails in zip(outputs[0],outputs[1],outputs[2]):
        # 抽取论元
        argus = set()
        entities[:, [0, -1]] -= np.inf
        entities[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(entities > threshold)):
            l: str = id2label[l].rsplit('+',1)
            argus.add((*l,h,t))
        # 构建链接
        links = set()
        for i1, (_,_, h1, t1) in enumerate(argus):
            for i2, (_,_, h2, t2) in enumerate(argus):
                if i2 > i1:
                    if heads[0, min(h1, h2), max(h1, h2)] > threshold and tails[0, min(t1, t2), max(t1, t2)] > threshold:
                        links.add((h1, t1, h2, t2))
                        links.add((h2, t2, h1, t1))
        # 析出事件
        events = []
        for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
            for event in clique_search(list(sub_argus), links):
                events.append([])
                for argu in event:
                    events[-1].append((label2id['+'.join(argu[:2])] , argu[2] - 1, argu[3] - 1))
                if trigger and all([argu[1] != u'触发词' for argu in event]):
                    events.pop()
        batch_result.append(events)
    return batch_result

class TransformerForGplinkerEvent(TransformerModel):
    def __init__(self,  *args, **kwargs):
        with_efficient = kwargs.pop('with_efficient',True)
        super(TransformerForGplinkerEvent, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, 1, 64, RoPE=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size, 1, 64, RoPE=False)

    def get_model_lr(self):
        return super(TransformerForGplinkerEvent, self).get_model_lr() + [
            (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        entity_labels: torch.Tensor = batch.pop('entity_labels', None)
        head_labels: torch.Tensor = batch.pop('head_labels', None)
        tail_labels: torch.Tensor = batch.pop('tail_labels', None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits1 = self.entities_layer(logits, attention_mask)
        logits2 = self.heads_layer(logits, attention_mask)
        logits3 = self.tails_layer(logits, attention_mask)
        if entity_labels is not None:
            loss1 = loss_for_gplinker(entity_labels, logits1)
            loss2 = loss_for_gplinker(head_labels, logits2)
            loss3 = loss_for_gplinker(tail_labels, logits3)
            loss = (loss1 + loss2 + loss3) / 3
            loss_dict = {'loss': loss,
                         'loss_entities': loss1,
                         'loss_head': loss2,
                         'loss_tail': loss3}
            outputs = (loss_dict, logits1, logits2, logits3,
                       entity_labels, head_labels, tail_labels)
        else:
            outputs = (logits1, logits2, logits3)
        return outputs



