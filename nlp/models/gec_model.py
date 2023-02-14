# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 9:53
import numpy as np
import torch
from torch import nn
from .transformer import TransformerModel
__all__ = [
    'TransformerForGec'
]
# [ [(action,position,vocab)]]
#抽取 action,position,vocab
def extract_gec(outputs,threshold=0):
    logits_action: np.ndarray
    logits_probs: np.ndarray
    result = []
    for logits_action,logits_probs,seqlen in zip(outputs[0],outputs[1],outputs[2]):
        logits_action = logits_action[1:seqlen-1]
        logits_probs = logits_probs[1:seqlen-1]
        ops = []
        for ids,action,probs in zip(range(seqlen),np.argmax(logits_action,axis=-1),np.argmax(logits_probs, axis=-1)):
            if logits_action[ids,action] < threshold:
                continue
            if action == 0:
                continue
            #add
            if action == 1:
                pass
            #delete
            elif action == 2:
                probs = 0
            #replace
            elif action == 3:
                pass
            else:
                raise ValueError('invalid action',action)

            ops.append((action, ids, probs))
        result.append(ops)
    return result

def extract_gec_from_labels(outputs):
    logits_action: np.ndarray
    logits_probs: np.ndarray
    result = []
    for logits_action,logits_probs,seqlen in zip(outputs[0],outputs[1],outputs[2]):
        logits_action = logits_action[1:seqlen-1]
        logits_probs = logits_probs[1:seqlen-1]
        ops = []
        for ids,action,probs in zip(range(seqlen),logits_action,logits_probs):
            if action == 0:
                continue
            if action == 0:
                continue
            #add
            if action == 1:
                pass
            #delete
            elif action == 2:
                probs = 0
            #replace
            elif action == 3:
                pass
            else:
                raise ValueError('invalid action',action)
            ops.append((action, ids, probs))
        result.append(ops)
    return result

class TransformerForGec(TransformerModel):
    def __init__(self, *args,**kwargs):
        super(TransformerForGec, self).__init__(*args,**kwargs)
        config = self.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 4)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()


    def get_model_lr(self):
        return super(TransformerForGec, self).get_model_lr() + [
            (self.classifier1, self.config.task_specific_params['learning_rate_for_task']),
            (self.classifier2, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def compute_loss(self, *args,**batch) -> tuple:
        labels_action: torch.Tensor = batch.pop('labels_action',None)
        labels_probs: torch.Tensor = batch.pop('labels_probs',None)
        attention_mask = batch['attention_mask']
        outputs = self.model(*args,**batch)
        logits = outputs[0]
        if self.model.training:
            logits = self.dropout(logits)
        logits1 = self.classifier1(logits)
        logits2 = self.classifier2(logits)

        logits_action = torch.softmax(logits1,dim=-1)
        logits_probs = torch.softmax(logits2,dim=-1)
        seqlens = torch.sum(attention_mask,dim=-1)
        if labels_action is not None:
            loss_action = self.loss_fn(logits1.view(-1,4),labels_action.view(-1))
            loss_probs = self.loss_fn(logits1.view(-1, self.config.num_labels),labels_probs.view(-1))
            loss = {
                'loss_action': loss_action,
                'loss_probs': loss_probs
            }
            outputs = (loss,logits_action,logits_probs,seqlens,labels_action,labels_probs)
        else:
            outputs = (logits_action,logits_probs,seqlens)
        return outputs

