# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import argparse
import typing
from typing import Any
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.auto.auto_factory import _get_model_class
from deep_training.data_helper import TrainingArguments, ModelArguments
from ..layers.mask import lm_mask,unilm_mask
from ..utils import configure_optimizers

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

class TransformerBase(PreTrainedModel):
    def __init__(self,config,*args,**kwargs):
        super(TransformerBase, self).__init__(config)
        self.config = config
        self.base_model_prefix = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def compute_loss(self, batch) -> tuple:
        return self.model(**batch)

    def post_init(self):
        return super(TransformerBase, self).post_init()

    def init_weights(self):
        return super(TransformerBase, self).init_weights()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config,model_args = None,None
        for item in args + tuple(kwargs.values()):
            if isinstance(item, PretrainedConfig):
                config = item
            if isinstance(item, ModelArguments):
                model_args = item

        args_new = tuple(item for item in args if
                      not isinstance(item, ModelArguments) and not isinstance(item, TrainingArguments) and not isinstance(item,
                                                                                                                    PretrainedConfig))
        kwargs_new = {k: v for k, v in kwargs.items() if
                      not isinstance(v, ModelArguments) and not isinstance(v, TrainingArguments) and not isinstance(v,
                                                                                                                    PretrainedConfig)}

        model_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        instance = super(TransformerBase, cls).from_pretrained(
            model_args.model_name_or_path,
            *args_new,
            **kwargs_new,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            **model_kwargs
        )
        return instance

    @property
    def model(self):
        if not self.base_model_prefix:
            return None
        return getattr(self, self.base_model_prefix,None)

    @model.setter
    def model(self, model):
        self.set_model(model)

    def set_model(self, model):
        keep_keys = [
            'config_class','load_tf_weights','base_model_prefix','supports_gradient_checkpointing','_init_weights','_set_gradient_checkpointing',
            '_keys_to_ignore_on_load_missing','_keys_to_ignore_on_load_unexpected','_no_split_modules','is_parallelizable','_shift_right','main_input_name',
            '_get_feat_extract_output_lengths','_get_feature_vector_attention_mask',#dummy_inputs
        ]
        for k in keep_keys:
            o = getattr(model,k,None)
            if o is None:
                continue
            setattr(self,k,o)
        setattr(self, self.base_model_prefix, model)

    def get_model_lr(self):
        return [(self.model, self.config.task_specific_params['learning_rate']), ] if hasattr(self, self.base_model_prefix) else []



class TransformerLightningModule(pl.LightningModule):
    def __init__(self, config, model_args: ModelArguments, training_args: TrainingArguments):
        super(TransformerLightningModule, self).__init__()
        if hasattr(config, 'task_specific_params') or config.task_specific_params is None:
            config.task_specific_params = {}
        task_specific_params = config.task_specific_params
        task_specific_params['learning_rate'] = training_args.learning_rate
        task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
            if training_args.learning_rate_for_task is not None else training_args.learning_rate
        print(training_args)
        print(model_args)
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.model_args = model_args
        self.training_args = training_args
        self.__model : TransformerBase

    @property
    def model(self) -> TransformerBase:
        return self.__model

    @model.setter
    def model(self,model):
        assert model is not None
        self.__model = model
        self.__model.log = self.log
        self.__model.log_dict = self.log_dict
        if hasattr(self.__model,'validation_epoch_end'):
            cname = self.validation_epoch_end.__qualname__
            if cname.endswith('.{}.validation_epoch_end'.format('LightningModule')) or cname.endswith('.{}.validation_epoch_end'.format('TransformerLightningModule')) :
                self.validation_epoch_end = self.__model.validation_epoch_end

    def get_model_lr(self):
        return self.model.get_model_lr()


    def compute_loss(self,batch):
        return self.model.compute_loss(batch)

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        return configure_optimizers(self.get_model_lr(), self.hparams,self.trainer.estimated_stepping_batches)

    def training_step(self, batch, batch_idx):
        outputs = self.compute_loss(batch)
        loss = outputs[0]

        if isinstance(loss,dict):
            self.log_dict(loss,prog_bar=True)
        else:
            self.log('loss',loss,prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.compute_loss(batch)
        loss = outputs[0]
        o = {}
        if isinstance(loss, dict):
            o = loss
            if 'loss' in o:
                o['val_loss'] = o.pop('loss')
        else:
            o['val_loss'] = loss.cpu().numpy()

        out = outputs[1:]
        if isinstance(out,(tuple,list)):
            o['outputs'] = [t.cpu().numpy() for t in out]
        else:
            o['outputs'] = [out.cpu().numpy()]
        return o

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.compute_loss(x)
        return [t.cpu().numpy() for t in outputs]






class TransformerModel(TransformerBase):
    def __init__(self, config,*args,**kwargs):
        super(TransformerModel, self).__init__(config,*args,**kwargs)
        config = self.config
        model = AutoModel.from_config(config)
        self.set_model(model)
        


class TransformerModelForUnilm(TransformerModel):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_model_lr(self):
        return super(TransformerModelForUnilm, self).get_model_lr() + \
               [(self.lm_head,self.config.task_specific_params['learning_rate_for_task']),]

    def compute_loss(self,batch):
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        if getattr(self.config, 'type_vocab_size', 0) != 2:
            batch.pop('token_type_ids')

        labels = batch.pop('labels',None)
        outputs = self(**batch)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,lm_logits,labels)
        else:
            outputs = (lm_logits,)
        return outputs



        

class TransformerForCausalLM(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForCausalLM.from_config(config)
        self.set_model(model)




class TransformerForMaskLM(TransformerBase):
    def __init__(self, config, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        config = self.config
        model = AutoModelForMaskedLM.from_config(config)
        self.set_model(model)




class TransformerForSeq2SeqLM(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForSeq2SeqLM.from_config(config)
        self.set_model(model)

        


class TransformerForSequenceClassification(TransformerBase):
    def __init__(self, config, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        config = self.config
        model = AutoModelForSequenceClassification.from_config(config)
        self.set_model(model)

        


class TransformerForQuestionAnswering(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForQuestionAnswering.from_config(config)
        self.set_model(model)

        


class TransformerForVisualQuestionAnswering(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForVisualQuestionAnswering.from_config(config)
        self.set_model(model)

        






class TransformerForTokenClassification(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForTokenClassification.from_config(config)
        self.set_model(model)
        

        


class TransformerForMultipleChoice(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForMultipleChoice.from_config(config)
        self.set_model(model)
        

        

class TransformerForNextSentencePrediction(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForNextSentencePrediction.from_config(config)
        self.set_model(model)


        


class TransformerForImageClassification(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForImageClassification.from_config(config)
        self.set_model(model)




class TransformerForImageSegmentation(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForImageSegmentation.from_config(config)
        self.set_model(model)
        

        



class TransformerForSemanticSegmentation(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model = AutoModelForSemanticSegmentation.from_config(config)
        self.set_model(model)
        

        

class TransformerForObjectDetection(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__( *args, **kwargs)
        config = self.config
        model =  AutoModelForObjectDetection.from_config(config)
        self.set_model(model)
        



class TransformerForAudioClassification(TransformerBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config = self.config
        model = AutoModelForAudioClassification.from_config(config)
        self.set_model(model)
        

        


class TransformerForMaskedImageModeling(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config = self.config
        model = AutoModelForMaskedImageModeling.from_config(config)
        self.set_model(model)



