# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import argparse
import copy
import sys
import typing
from typing import Any, Callable, cast, Dict, IO, MutableMapping, Optional, Type, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.auto.auto_factory import _get_model_class

from ..losses.lm_loss import LM_loss
from ...data_helper import TrainingArguments, ModelArguments, PrefixModelArguments, DataArguments
from ..layers.mask import lm_mask,unilm_mask
from ..utils import configure_optimizers, get_value_from_args
from ..utils.adversarial import AdversarialMethods


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



def verify_manual_optimization_support(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    if model.automatic_optimization:
        return
    trainer.gradient_clip_val = None
    trainer.accumulate_grad_batches = 1


class MyLightningModule(pl.LightningModule):
    def __init__(self,*args: Any, **kwargs: Any):
        super(MyLightningModule, self).__init__(*args,**kwargs)

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: typing.Union[str, IO],
            map_location = None,
            hparams_file: typing.Optional[str] = None,
            strict: bool = True,
            **kwargs: Any,
    ) -> typing.Union["pl.LightningModule", "pl.LightningDataModule","MyLightningModule"]:
        return super(MyLightningModule, cls).load_from_checkpoint(checkpoint_path,map_location,hparams_file,strict,**kwargs)

    @property
    def backbone(self):
        return self.__model

    @property
    def model(self):
        return self.__model

    def convert_to_onnx(self, file_path,
                        input_sample=(
                                ("input_ids",torch.ones(size=(1, 128), dtype=torch.int64)),
                                ("attention_mask",torch.ones(size=(1, 128), dtype=torch.int64)),
                        ),
                        input_names=("input_ids", "attention_mask"),
                        output_names=("pred_ids",),
                        dynamic_axes=None or {"input_ids": [0, 1], "attention_mask": [0, 1], "pred_ids": [0, 1]},
                        opset_version=14,
                        verbose=True,
                        do_constant_folding=True
                        ):
        self.eval()
        self.to('cuda')
        self.to_onnx(file_path,
                     input_sample=input_sample,
                     verbose=verbose,
                     opset_version=opset_version,
                     do_constant_folding=do_constant_folding,
                     input_names=input_names,
                     output_names=output_names,
                     dynamic_axes=dynamic_axes)



class TransformerFakeMeta(type):
    def __new__(cls, name, base, attr,*args,**kwargs):
        base_new = tuple(b for b in base if b != MyLightningModule)
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


class TransformerBase(MyLightningModule,metaclass=TransformerFakeMeta):
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



class TransformerLightningModule(MyLightningModule):
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
        if training_args.adv['mode'] != None:
            assert training_args.adv['mode']  in AdversarialMethods.keys(), ValueError('no support adv mode {} , must be in {}'.format(training_args.adv['mode'],','.join(AdversarialMethods.keys())))
            self.automatic_optimization = False

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


        self.training_step_fn = self.training_step
        self.embeddings_forward_fn = None
        if training_args.adv['mode'] is not None:
            self.embeddings_forward_fn = self.model.model.embeddings.forward

            self.training_step = self.adv_training_step
            if training_args.adv['mode'].find('local') != -1:
                self.adversarial = AdversarialMethods[training_args.adv['mode']](model=self.model)
            else:
                self.adversarial = AdversarialMethods[training_args.adv['mode']](model=self.model,
                                                                                 emb_name=training_args.adv.get('emb_name', 'embedding'))
            k = 'pytorch_lightning.trainer.configuration_validator'
            if k in sys.modules:
                setattr( sys.modules[k],'__verify_manual_optimization_support' , verify_manual_optimization_support)
        else:
            self.adversarial = None

        self.gradient_clip_val = training_args.max_grad_norm


    @property
    def backbone(self):
        return self.__model

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
        kwargs.update(dict(args))
        return self.model.compute_loss(**kwargs)


    def forward(self,*args, **kwargs):
        kwargs.update(dict(args))
        return self.compute_loss(**kwargs)


    def setup(self, stage: str) -> None:
        setattr(self.__model, 'trainer', self.trainer)
        setattr(self.__model, 'estimated_stepping_batches', self.trainer.estimated_stepping_batches)


    def configure_optimizers(self):
        return configure_optimizers(self.get_model_lr(), self.training_args,self.trainer.estimated_stepping_batches)


    def manual_backward(self,loss: Tensor, *args: Any, **kwargs: Any):
        if isinstance(loss,dict):
            loss = loss['loss']
        return super(TransformerLightningModule, self).manual_backward(loss)



    def adv_training_step(self,batch):
        mode = self.training_args.adv['mode']
        opt = self.optimizers()
        gradient_clip_val = self.gradient_clip_val
        epsilon = self.training_args.adv['epsilon']
        if mode == 'fgm':
            opt.zero_grad()
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)
            self.adversarial.attack(epsilon=epsilon)
            loss = self.training_step_fn(batch)
            opt.zero_grad()
            self.manual_backward(loss)
            if gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            opt.step()
            self.adversarial.restore()  # 恢复embedding参数
            self.model.zero_grad()
        elif mode == 'fgsm_local':
            alpha = self.training_args.adv['alpha']
            opt.zero_grad()
            delta = torch.zeros((*batch['input_ids'].size()[:2], self.config.hidden_size), dtype=torch.float32).to(
                batch['input_ids'].device)
            def forward_fn(*args, **kwargs):
                embedding_output = self.embeddings_forward_fn(*args, **kwargs)
                embedding_output += delta
                return embedding_output
            setattr(self.model.model.embeddings, 'forward', forward_fn)
            delta = self.adversarial.attack(is_first_attack=True, delta=delta,alpha=alpha,epsilon=epsilon)
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)

            delta = self.adversarial.attack(delta=delta,alpha=alpha,epsilon=epsilon)
            loss = self.training_step_fn(batch)
            opt.zero_grad()
            self.manual_backward(loss)
            if gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            opt.step()
            self.model.zero_grad()

            setattr(self.model.model.embeddings, 'forward', self.embeddings_forward_fn)
        elif mode == 'fgsm':
            alpha = self.training_args.adv['alpha']
            self.adversarial.attack(is_first_attack=True,alpha=alpha,epsilon=epsilon)
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)

            self.adversarial.attack(alpha=alpha,epsilon=epsilon)
            loss = self.training_step_fn(batch)
            opt.zero_grad()
            self.manual_backward(loss)
            if gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            opt.step()
            self.adversarial.restore()  # 恢复embedding参数
            self.model.zero_grad()
        elif mode == 'pgd':
            alpha = self.training_args.adv['alpha']
            opt.zero_grad()
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)

            self.adversarial.backup_grad()
            attack_iters = self.training_args.adv['attack_iters']
            for t in range(attack_iters):
                self.adversarial.attack(is_first_attack=(t == 0),alpha=alpha,epsilon=epsilon)
                if t != attack_iters - 1:
                    opt.zero_grad()
                else:
                    self.adversarial.restore_grad()
                loss = self.training_step_fn(batch)
                self.manual_backward(loss)
                if gradient_clip_val is not None:
                    self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            self.adversarial.restore()  # 恢复embedding参数
            opt.step()
            self.model.zero_grad()
        elif mode == 'free_local':
            if not hasattr(self.adversarial,'delta_'):
                setattr(self.adversarial,'delta_', torch.zeros((batch['input_ids'].size(0),self.config.max_position_embeddings, self.config.hidden_size),requires_grad=True).to(batch['input_ids'].device))
            delta = getattr(self.adversarial,'delta_')
            def forward_fn(*args, **kwargs):
                embedding_output = self.embeddings_forward_fn(*args, **kwargs)
                embedding_output += delta[:embedding_output.size(0),:embedding_output.size(1)]
                return embedding_output

            setattr(self.model.model.embeddings, 'forward', forward_fn)
            for _ in range(self.training_args.adv['minibatch_replays']):
                delta.retain_grad()
                loss = self.training_step_fn(batch)
                opt.zero_grad()
                self.manual_backward(loss)
                if gradient_clip_val is not None:
                    self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
                opt.step()
                delta = self.adversarial.attack(delta=delta,epsilon=epsilon)
                # delta.grad.zero_()
                self.model.zero_grad()

            setattr(self.model.model.embeddings, 'forward', self.embeddings_forward_fn)
        elif mode == 'free':
            for _ in range(self.training_args.adv['minibatch_replays']):
                opt.zero_grad()
                loss = self.training_step_fn(batch)
                self.manual_backward(loss)
                if gradient_clip_val is not None:
                    self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
                opt.step()
                self.adversarial.attack(epsilon=epsilon)
                self.model.zero_grad()
        else:
            opt.zero_grad()
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)
            if gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            opt.step()
            self.model.zero_grad()
        return loss

    def training_step(self, batch):
        if isinstance(batch, dict):
            outputs = self.compute_loss(**batch)
        else:
            outputs = self.compute_loss(**dict(batch))
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
            outputs = self.compute_loss(**dict(batch))

        loss = outputs[0]
        o = {}
        if loss is not None:
            if isinstance(loss, dict):
                o = loss
                if 'loss' in o:
                    o['val_loss'] = o.pop('loss')
            else:
                o['val_loss'] = loss.cpu().detach().numpy()

        out = outputs[1:]
        if isinstance(out,(tuple,list)):
            o['outputs'] = []
            obj = o['outputs']
            for t in out:
                if t is None:
                    obj.append(t)
                elif isinstance(t,torch.Tensor):
                    obj.append(t.cpu().detach().numpy())
                elif isinstance(t, list) or isinstance(t, tuple):
                    tmp_list =[_ for _ in t]
                    for idx in range(len(tmp_list)):
                        node = tmp_list[idx]
                        if isinstance(node, torch.Tensor):
                            tmp_list[idx] = node.cpu().detach().numpy()
                        elif isinstance(node, list) or isinstance(node, tuple):
                            tmp_list[idx] = [_.cpu().detach().numpy() for _ in node]
                        else:
                            raise ValueError('validation_step: outputs not support', type(t))
                    obj.append(tmp_list)
                elif isinstance(t, dict):
                    obj.append({k:v.cpu().detach().numpy() for k,v in t.items()})
                else:
                    raise ValueError('validation_step: outputs not support', type(t))
        else:
            o['outputs'] = out.cpu().detach().numpy()
        return o

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            outputs = self.compute_loss(**batch)
        else:
            outputs = self.compute_loss(**dict(batch))
        o = {}
        out = outputs
        if isinstance(out, (tuple, list)):
            o['outputs'] = []
            obj = o['outputs']
            for t in out:
                if t is None:
                    obj.append(t)
                elif isinstance(t, torch.Tensor):
                    obj.append(t.cpu().detach().numpy())
                elif isinstance(t, list) or isinstance(t, tuple):
                    tmp_list =[_ for _ in t]
                    for idx in range(len(tmp_list)):
                        node = tmp_list[idx]
                        if isinstance(node,torch.Tensor):
                            tmp_list[idx] = node.cpu().detach().numpy()
                        elif isinstance(node, list) or isinstance(node, tuple):
                            tmp_list[idx] = [_.cpu().detach().numpy() for _ in node]
                        else:
                            raise ValueError('test_step: outputs not support', type(t))
                    obj.append(tmp_list)
                elif isinstance(t, dict):
                    obj.append({k: v.cpu().detach().numpy() for k, v in t.items()})
                else:
                    raise ValueError('test_step: outputs not support',type(t))
        else:
            o['outputs'] = out.cpu().detach().numpy()
        return o




class TransformerModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerModel, self).__init__(*args,**kwargs)
        self.set_model( self.from_pretrained(AutoModel,*args,**kwargs))
        


class TransformerModelForUnilm(TransformerModel):
    def __init__(self,*args: Any, **kwargs: Any):
        ignore_index = kwargs.pop('ignore_index',-100)
        super().__init__(*args, **kwargs)
        self.loss_fct = LM_loss(ignore_index=ignore_index)
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
            loss = self.loss_fct(lm_logits,labels)
            outputs = (loss,lm_logits,labels)
        else:
            outputs = (lm_logits,)
        return outputs



        

class TransformerForCausalLM(TransformerBase):
    def __init__(self,*args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForCausalLM, *args, **kwargs))


    # def compute_loss(self, *args,**batch) -> tuple:
    #     outputs = super(TransformerForCausalLM, self).compute_loss(*args,**batch)
    #     if not self.model.training:
    #         if 'labels' in batch:
    #             outputs = (*outputs[:2],)
    #         else:
    #             outputs = (outputs[0],)
    #     return outputs


class TransformerForMaskLM(TransformerBase):
    def __init__(self,  *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.set_model(self.from_pretrained(AutoModelForMaskedLM, *args, **kwargs))

    # def compute_loss(self, *args,**batch) -> tuple:
    #     outputs = super(TransformerForMaskLM, self).compute_loss(*args,**batch)
    #     if not self.model.training:
    #         if 'labels' in batch:
    #             outputs = (*outputs[:2],)
    #         else:
    #             outputs = (outputs[0],)
    #     return outputs



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



