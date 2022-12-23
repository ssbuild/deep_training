# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import argparse
import typing
from typing import Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.auto.auto_factory import _get_model_class
from ...data_helper import TrainingArguments, ModelArguments, PrefixModelArguments, DataArguments
from ..layers.mask import lm_mask,unilm_mask
from ..utils import configure_optimizers, get_value_from_args

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
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
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    get_linear_schedule_with_warmup, AutoModelForPreTraining, AutoModel, PreTrainedModel
)

# class TransformerMeta(type):
#     def __new__(cls, name, base, attr,*args,**kwargs):
#         alter = tuple(b for b in base if issubclass(b,TransformerBase))
#         cls_ = super(TransformerMeta, cls).__new__(cls, name, (TransformerLightningModule,) + tuple(b for b in base if not issubclass(b, TransformerBase)), attr)
#         cls_.__ALTER_CLASS__ = alter
#         return cls_

class TransformerFakeMeta(type):
    def __new__(cls, name, base, attr,*args,**kwargs):
        base_new =  tuple(b for b in base if b != pl.LightningModule)
        if name == 'TransformerBase':
            base_new =  base_new + (nn.Module,)
        with_pl = kwargs.get('with_pl',False)
        alter = None
        if with_pl:
            alter = tuple(b for b in base if issubclass(b, TransformerBase))
            base_new = (TransformerLightningModule,) + tuple(b for b in base_new if not issubclass(b, TransformerBase))
        cls_ = super(TransformerFakeMeta, cls).__new__(cls, name, base_new, attr)
        if alter is not None:
            cls_.__ALTER_CLASS__ = alter
        return cls_


class TransformerBase(pl.LightningModule,metaclass=TransformerFakeMeta):
    def __init__(self,*args,**kwargs):
        config = get_value_from_args('config',PretrainedConfig,*args,**kwargs)
        super(TransformerBase, self).__init__()
        self.config = config
        self.base_model_prefix = None
        self._trainer:  typing.Optional["pl.Trainer"]  = None

    def forward(self, *args, **batch):
        return self.model(*args,**batch)

    def compute_loss(self, *args,**batch) -> tuple:
        return self.model(*args,**batch)

    def post_init(self):
        return self.model.post_init()

    def init_weights(self):
        return self.model.init_weights()

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self,trainer: typing.Optional["pl.Trainer"]):
         self._trainer = trainer

    @property
    def current_epoch(self):
        return self.trainer.current_epoch if self._trainer else 0

    @property
    def global_step(self):
        return self.trainer.global_step if self._trainer else 0

    @property
    def max_epochs(self) -> typing.Optional[int]:
        return self.trainer.max_epochs if self._trainer else 0

    @property
    def min_epochs(self) -> typing.Optional[int]:
        return self.trainer.min_epochs if self._trainer else 0

    @property
    def max_steps(self) -> int:
        return self.trainer.max_steps if self._trainer else 0

    @property
    def min_steps(self) -> int:
        return self.trainer.min_steps if self._trainer else 0


    def from_pretrained(self,CLS, *args, **kwargs):
        config = get_value_from_args('config', PretrainedConfig, *args, **kwargs)
        model_args = get_value_from_args('model_args', ModelArguments, *args, **kwargs)

        if model_args.model_name_or_path:
            args_new = tuple(v for v in args
                             if not isinstance(v, ModelArguments) and \
                             not isinstance(v, TrainingArguments) and \
                             not isinstance(v,PretrainedConfig) and \
                             not isinstance(v,PrefixModelArguments) and \
                             not isinstance(v,DataArguments)
                             )
            kwargs_new = {k: v for k, v in kwargs.items()
                          if not isinstance(v, ModelArguments) and \
                          not isinstance(v, TrainingArguments) and \
                          not isinstance(v, PretrainedConfig) and \
                          not isinstance(v, PrefixModelArguments) and \
                          not isinstance(v, DataArguments)
                          }

            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            cls_ = CLS.from_pretrained(
                model_args.model_name_or_path,
                *args_new,
                **kwargs_new,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            cls_ = CLS.from_config(config)
            cls_.post_init()
        return cls_

    @property
    def model(self):
        if not self.base_model_prefix:
            return None
        return getattr(self, self.base_model_prefix,None)

    @model.setter
    def model(self, model):
        self.set_model(model)

    def set_model(self, model):
        # keep_keys = [
        #     'config_class','load_tf_weights','base_model_prefix','supports_gradient_checkpointing','_init_weights','_set_gradient_checkpointing',
        #     '_keys_to_ignore_on_load_missing','_keys_to_ignore_on_load_unexpected','_no_split_modules','is_parallelizable','_shift_right','main_input_name',
        #     '_get_feat_extract_output_lengths','_get_feature_vector_attention_mask',#dummy_inputs
        # ]
        keep_keys = ['config_class','base_model_prefix']
        for k in keep_keys:
            o = getattr(model,k,None)
            if o is None:
                continue
            setattr(self,k,o)

        setattr(self, self.base_model_prefix, model)

    def get_model_lr(self):
        return [(self.model, self.config.task_specific_params['learning_rate']), ] if hasattr(self, self.base_model_prefix) else []



class TransformerLightningModule(pl.LightningModule):
    def __init__(self, *args,**kwargs):
        config = get_value_from_args('config',PretrainedConfig,*args,**kwargs)
        model_args = get_value_from_args('model_args', ModelArguments, *args, **kwargs)
        training_args = get_value_from_args('training_args', TrainingArguments, *args, **kwargs)
        super(TransformerLightningModule, self).__init__()
        if not hasattr(config, 'task_specific_params') or config.task_specific_params is None:
            config.task_specific_params = {}
        task_specific_params = config.task_specific_params
        task_specific_params['learning_rate'] = training_args.learning_rate
        task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
            if training_args.learning_rate_for_task is not None else training_args.learning_rate
        print(training_args)
        print(model_args)
        try:
            self.save_hyperparameters(ignore=['config'])
        except:
            pass
        self.config = config
        self.model_args = model_args
        self.training_args = training_args
        self.__model : typing.Optional[TransformerBase] = None
        if hasattr(self,'__ALTER_CLASS__') and len(self.__ALTER_CLASS__) > 0:
            self.set_model(self.__ALTER_CLASS__[0](*args, **kwargs))

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.set_model(model)

    def set_model(self, model):
        assert model is not None
        self.__model = model

        copy_attr = [
            'log','log_dict'
        ]
        for k in copy_attr:
            setattr(self.__model,k,getattr(self,k))
        # setattr(self.__model, 'estimated_stepping_batches', self.trainer.estimated_stepping_batches)
        # if hasattr(self.__model,'validation_epoch_end'):
        #     cname = self.validation_epoch_end.__qualname__
        #     if cname.endswith('.{}.validation_epoch_end'.format('LightningModule')) or cname.endswith('.{}.validation_epoch_end'.format('TransformerLightningModule')) :
        #         self.validation_epoch_end = self.__model.validation_epoch_end

        event_ = [
            'training_step',
            'training_step_end',
            'training_epoch_end',
            'validation_step',
            'validation_step_end',
            'validation_epoch_end',
            'test_step',
            'test_step_end',
            'test_epoch_end',
            'predict_step',
            'configure_optimizers',
            'configure_gradient_clipping',
            'lr_scheduler_step',
            'optimizer_step',
            'optimizer_zero_grad',
        ]
        for e in event_:
            a = getattr(self.__model, e,None)
            if a is not None:
                setattr(self,e,a)




    def get_model_lr(self):
        return self.model.get_model_lr()


    def compute_loss(self,*args, **kwargs):
        return self.model.compute_loss(*args, **kwargs)


    def forward(self,*args, **kwargs):
        return self.compute_loss(*args,**kwargs)


    def setup(self, stage: str) -> None:
        setattr(self.__model, 'trainer', self.trainer)
        setattr(self.__model, 'estimated_stepping_batches', self.trainer.estimated_stepping_batches)


    def configure_optimizers(self):
        return configure_optimizers(self.get_model_lr(), self.training_args,self.trainer.estimated_stepping_batches)

    def training_step(self, batch):
        if isinstance(batch, dict):
            outputs = self.compute_loss(**batch)
        else:
            outputs = self.compute_loss(*batch)

        loss = outputs[0]
        if isinstance(loss,dict):
            self.log_dict(loss,prog_bar=True)
        else:
            self.log('loss',loss,prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            outputs = self.compute_loss(**batch)
        else:
            outputs = self.compute_loss(*batch)

        loss = outputs[0]
        o = {}
        if loss is not None:
            if isinstance(loss, dict):
                o = loss
                if 'loss' in o:
                    o['val_loss'] = o.pop('loss')
            else:
                o['val_loss'] = loss.cpu().numpy()

        out = outputs[1:]
        if isinstance(out,(tuple,list)):
            o['outputs'] = []
            obj = o['outputs']
            for t in out:
                if t is None:
                    obj.append(t)
                elif isinstance(t,torch.Tensor):
                    obj.append(t.cpu().numpy())
                elif isinstance(t, list):
                    obj.append([tt.cpu().numpy() for tt in t])
                elif isinstance(t, dict):
                    obj.append({k:v.cpu().numpy() for k,v in t.items()})
                else:
                    raise ValueError('not support')
        else:
            o['outputs'] = out.cpu().numpy()
        return o

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            outputs = self.compute_loss(**batch)
        else:
            outputs = self.compute_loss(*batch)

        o = {}
        out = outputs
        if isinstance(out, (tuple, list)):
            o['outputs'] = []
            obj = o['outputs']
            for t in out:
                if t is None:
                    obj.append(t)
                elif isinstance(t, torch.Tensor):
                    obj.append(t.cpu().numpy())
                elif isinstance(t, list):
                    obj.append([tt.cpu().numpy() for tt in t])
                elif isinstance(t, dict):
                    obj.append({k: v.cpu().numpy() for k, v in t.items()})
                else:
                    raise ValueError('not support')
        else:
            o['outputs'] = out.cpu().numpy()
        return o




class TransformerModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerModel, self).__init__(*args,**kwargs)
        self.set_model( self.from_pretrained(AutoModel,*args,**kwargs))
        


class TransformerModelForUnilm(TransformerModel):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def get_model_lr(self):
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
            labels = labels.long()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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



