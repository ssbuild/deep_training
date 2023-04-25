# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import sys
import typing
from functools import partial
from typing import Any, IO
import torch
from torch import nn, Tensor
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForVisualQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForSemanticSegmentation,
    AutoModelForObjectDetection,
    AutoModelForAudioClassification,
    AutoModelForMaskedImageModeling,
    AutoModelForCausalLM,
    PretrainedConfig,
    AutoModel
)

from .transformer_base import TransformerBase,TransformerLightningModule,MyLightningModule
from ..layers.mask import unilm_mask
from ..losses.lm_loss import LM_loss



class TransformerModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(AutoModel,*args,**kwargs))
        


class TransformerModelForUnilm(TransformerModel):
    def __init__(self,*args: Any, **kwargs: Any):
        ignore_index = kwargs.pop('ignore_index',-100)
        super().__init__(*args, **kwargs)
        self.loss_fct = LM_loss(ignore_index=ignore_index)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def get_model_lr(self,*args,**kwargs):
        return super(TransformerModelForUnilm, self).get_model_lr() + \
               [(self.lm_head,self.config.task_specific_params['learning_rate_for_task']),]

    def compute_loss(self, *args,**batch) -> tuple:
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        if getattr(self.config, 'type_vocab_size', 0) != 2:
            batch.pop('token_type_ids')

        labels = batch.pop('labels',None)
        outputs = self.model(*args,**batch)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.loss_fct(lm_logits,labels)
            outputs = (loss,lm_logits,labels)
        else:
            outputs = (lm_logits,)
        return outputs



        

class TransformerForCausalLM(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForCausalLM, *args, **kwargs))




class TransformerForMaskLM(TransformerBase):
    def __init__(self,  *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForMaskedLM, *args, **kwargs))




class TransformerForSeq2SeqLM(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForSeq2SeqLM, *args, **kwargs))


    def compute_loss(self, *args,**batch) -> tuple:
        outputs = super(TransformerForSeq2SeqLM, self).compute_loss(*args,**batch)
        if not self.model.training:
            if 'labels' in batch:
                outputs = (*outputs[:2],)
            else:
                outputs = (outputs[0],)
        return outputs



class TransformerForSequenceClassification(TransformerBase):
    def __init__(self,  *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForSequenceClassification, *args, **kwargs))

        


class TransformerForQuestionAnswering(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForQuestionAnswering, *args, **kwargs))

        


class TransformerForVisualQuestionAnswering(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForVisualQuestionAnswering, *args, **kwargs))

        

class TransformerForTokenClassification(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForTokenClassification, *args, **kwargs))
        

class TransformerForMultipleChoice(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForMultipleChoice, *args, **kwargs))
        

class TransformerForNextSentencePrediction(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForNextSentencePrediction, *args, **kwargs))


        


class TransformerForImageClassification(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForImageClassification, *args, **kwargs))




class TransformerForImageSegmentation(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForImageSegmentation, *args, **kwargs))
        


class TransformerForSemanticSegmentation(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForSemanticSegmentation, *args, **kwargs))
        

class TransformerForObjectDetection(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForObjectDetection, *args, **kwargs))
        



class TransformerForAudioClassification(TransformerBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForAudioClassification, *args, **kwargs))
        

        


class TransformerForMaskedImageModeling(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForMaskedImageModeling, *args, **kwargs))



