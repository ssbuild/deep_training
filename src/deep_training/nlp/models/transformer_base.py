# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 14:35
import dataclasses
import sys
from functools import partial
from typing import Any, IO, Union, Optional, Dict
import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.apply_func import apply_to_collection
from torch import nn, Tensor
from transformers import (
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils import configure_optimizers, get_value_from_args_assert, get_value_from_args
from ..utils.adversarial import AdversarialMethods
from ...data_helper import TrainingArguments, ModelArguments, PrefixModelArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC


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
            checkpoint_path: Union[str, IO],
            map_location = None,
            hparams_file: Optional[str] = None,
            strict: bool = True,
            **kwargs: Any,
    ) -> Union["pl.LightningModule", "pl.LightningDataModule","MyLightningModule"]:
        return super(MyLightningModule, cls).load_from_checkpoint(checkpoint_path,map_location,hparams_file,strict,**kwargs)

    @property
    def backbone(self) -> Union[nn.Module,Any]:
        return self.__model

    def set_backbone(self, model , copy_attr=True):
        ...

    def convert_to_onnx(self, file_path,
                        input_sample=(
                                ("input_ids",torch.ones(size=(1, 128), dtype=torch.int32)),
                                ("attention_mask",torch.ones(size=(1, 128), dtype=torch.int32)),
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
            base_new = base_new + (nn.Module,)
        with_pl = kwargs.get('with_pl',False)
        backbone_class = None
        if with_pl:
            backbone_class = tuple(b for b in base if issubclass(b, TransformerBase))
            base_new = (TransformerLightningModule,) + tuple(b for b in base_new if not issubclass(b, TransformerBase))
        cls_ = super(TransformerFakeMeta, cls).__new__(cls, name, base_new, attr)
        if backbone_class is not None:
            cls_.__BACKBONE_CLASS__ = backbone_class
        return cls_



class TransformerBase(MyLightningModule, metaclass=TransformerFakeMeta):
    def __init__(self,*args,**kwargs):
        config = get_value_from_args_assert('config',PretrainedConfig,*args,**kwargs)
        super(TransformerBase, self).__init__()
        self.config = config
        self.base_model_prefix = None
        self.config_class = None
        self._trainer:  Optional["pl.Trainer"] = None

    def forward(self, *args, **batch):
        return self.model(*args, **batch)

    def compute_loss(self, *args, **batch) -> tuple:
        return self.model(*args, **batch)

    def post_init(self):
        return self.model.post_init()

    def init_weights(self):
        return self.model.init_weights()

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self,trainer: Optional["pl.Trainer"]):
         self._trainer = trainer

    @property
    def current_epoch(self):
        return self.trainer.current_epoch if self._trainer else 0

    @property
    def global_step(self):
        return self.trainer.global_step if self._trainer else 0

    @property
    def max_epochs(self) -> Optional[int]:
        return self.trainer.max_epochs if self._trainer else 0

    @property
    def min_epochs(self) -> Optional[int]:
        return self.trainer.min_epochs if self._trainer else 0

    @property
    def max_steps(self) -> int:
        return self.trainer.max_steps if self._trainer else 0

    @property
    def min_steps(self) -> int:
        return self.trainer.min_steps if self._trainer else 0


    def from_pretrained(self,CLS, *args, **kwargs):
        config = get_value_from_args_assert('config', PretrainedConfig, *args, **kwargs)
        model_args = get_value_from_args_assert('model_args', ModelArguments, *args, **kwargs)

        if model_args.model_name_or_path:
            args_new = tuple(v for v in args
                             if not isinstance(v, ModelArguments) and \
                             not isinstance(v, TrainingArguments) and \
                             not isinstance(v, TrainingArgumentsHF) and \
                             not isinstance(v, TrainingArgumentsCL) and \
                             not isinstance(v, TrainingArgumentsAC) and \
                             not isinstance(v,PretrainedConfig) and \
                             not isinstance(v,PrefixModelArguments) and \
                             not isinstance(v,DataArguments)
                             )
            kwargs_new = {k: v for k, v in kwargs.items()
                          if not isinstance(v, ModelArguments) and \
                          not isinstance(v, TrainingArguments) and \
                          not isinstance(v, TrainingArgumentsHF) and \
                          not isinstance(v, TrainingArgumentsCL) and \
                          not isinstance(v, TrainingArgumentsAC) and \
                          not isinstance(v, PretrainedConfig) and \
                          not isinstance(v, PrefixModelArguments) and \
                          not isinstance(v, DataArguments)
                          }

            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision":  model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
                **kwargs_new,
            }
            cls_ = CLS.from_pretrained(
                model_args.model_name_or_path,
                *args_new,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        elif hasattr(CLS,'from_config'):
            cls_ = CLS.from_config(config)
            cls_.post_init()
        elif hasattr(CLS, '_from_config'):
            cls_ = CLS._from_config(config)
            cls_.post_init()
        else:
            cls_ = CLS(config)
            cls_.post_init()
        return cls_

    @property
    def model(self):
        if not self.base_model_prefix:
            return None
        return getattr(self, self.base_model_prefix,None)

    @model.setter
    def model(self, model , copy_attr=True):
        self.set_model(model,copy_attr=copy_attr)

    def set_model(self, model , copy_attr=True):
        if copy_attr:
            keep_keys = ['config_class','base_model_prefix']
            for k in keep_keys:
                o = getattr(model,k,None)
                if o is None:
                    continue
                if o == 'model':
                    o = 'model_'
                setattr(self,k,o)

        assert self.base_model_prefix is not None, ValueError('base_model_prefix is not allow empty')
        setattr(self, self.base_model_prefix, model)

    def get_model_lr(self,model=None,lr=None):
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if model is not None:
            return [(model,lr)]
        # return [(self.model if self.base_model_prefix is not None else self , lr), ]
        return [(self, lr), ]

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

class TransformerLightningModule(MyLightningModule):
    def __init__(self, *args,**kwargs):
        config = get_value_from_args_assert('config',PretrainedConfig,*args,**kwargs)
        model_args = get_value_from_args_assert('model_args', ModelArguments, *args, **kwargs)
        training_args = get_value_from_args('training_args', (TrainingArguments,TrainingArgumentsHF,TrainingArgumentsCL,TrainingArgumentsAC), *args, **kwargs)

        super(TransformerLightningModule, self).__init__()
        if not hasattr(config, 'task_specific_params') or config.task_specific_params is None:
            config.task_specific_params = {}
        task_specific_params = config.task_specific_params
        if training_args is not None:
            self.gradient_clip_val = training_args.max_grad_norm
            task_specific_params['learning_rate'] = training_args.learning_rate
            task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
                if training_args.learning_rate_for_task is not None else training_args.learning_rate

            if isinstance(training_args,TrainingArguments):
                if training_args.adv is not None and training_args.adv['mode'] != None:
                    assert training_args.adv['mode']  in AdversarialMethods.keys(), ValueError('no support adv mode {} , must be in {}'.format(training_args.adv['mode'],','.join(AdversarialMethods.keys())))
                    self.automatic_optimization = False

        self.config = config
        self.model_args = model_args
        self.training_args = training_args
        try:
            if training_args is None or isinstance(training_args, TrainingArguments) or  isinstance(training_args, TrainingArgumentsCL):
                print(config)
                print(training_args)
            print(model_args)

            self.save_hyperparameters(ignore=['config','torch_dtype','quantization_config'])
        except:
            pass

        self.transformer_base : Optional[TransformerBase] = None
        if hasattr(self,'__BACKBONE_CLASS__') and len(self.__BACKBONE_CLASS__) > 0:
            self.set_model(self.__BACKBONE_CLASS__[0](*args, **kwargs))

        self.training_step_fn = self.training_step
        self.embeddings_forward_fn = None

        self.hook_adv()
        self.hook_hierarchical_position()


    def hook_adv(self):
        training_args = self.training_args
        if training_args is None or not isinstance(training_args, TrainingArguments):
            return
            # 对抗训练
        if training_args.adv is not None and training_args.adv['mode'] is not None:
            self.embeddings_forward_fn = self.get_embeddings_module().embeddings.forward

            self.training_step = self.adv_training_step
            if training_args.adv['mode'].find('local') != -1:
                self.adversarial = AdversarialMethods[training_args.adv['mode']](model=self.model)
            else:
                self.adversarial = AdversarialMethods[training_args.adv['mode']](model=self.model,
                                                                                 emb_name=training_args.adv.get(
                                                                                     'emb_name', 'embedding'))

            k = 'lightning.pytorch.trainer.configuration_validator'
            if k in sys.modules:
                setattr(sys.modules[k], '__verify_manual_optimization_support', verify_manual_optimization_support)
        else:
            self.adversarial = None

    def hook_hierarchical_position(self):
        training_args = self.training_args
        if training_args is None or not isinstance(training_args,TrainingArguments):
            return

        if training_args.hierarchical_position is not None and (
                training_args.hierarchical_position > 0 and training_args.hierarchical_position < 1):
            # 绝对位置编码 分层位置编码
            def forward(cls, input: Tensor) -> Tensor:
                # return F.embedding(
                #     input, self.weight, self.padding_idx, self.max_norm,
                #     self.norm_type, self.scale_grad_by_freq, self.sparse)
                position_ids = input
                alpha = training_args.hierarchical_position
                embeddings = cls.weight - alpha * cls.weight[:1]
                embeddings = embeddings / (1 - alpha)
                x_idx = position_ids // cls.num_embeddings
                y_idx = position_ids % cls.num_embeddings

                embeddings_x = torch.index_select(embeddings, dim=0, index=x_idx.view(-1))
                embeddings_y = torch.index_select(embeddings, dim=0, index=y_idx.view(-1))
                embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
                return embeddings

            position_embeddings = self.get_embeddings_module().embeddings.position_embeddings
            position_embeddings.forward = partial(forward, position_embeddings)

    def get_embeddings_module(self):
        base_model_prefix = self.backbone.base_model_prefix
        current_model = self.backbone.model
        tmp_obj = current_model
        while tmp_obj is not None:
            if hasattr(tmp_obj, 'embeddings'):
                return tmp_obj
            current_model = tmp_obj
            tmp_obj = getattr(current_model, base_model_prefix, None)
        return tmp_obj

    @property
    def backbone(self) -> nn.Module:
        return self.transformer_base

    @backbone.setter
    def backbone(self,model , copy_attr=True):
        self.set_backbone(model=model,copy_attr=copy_attr)

    def set_backbone(self, model , copy_attr=True):
        self.set_model(model,copy_attr=copy_attr)

    @property
    def model(self) -> nn.Module:
        return self.transformer_base

    @model.setter
    def model(self, model, copy_attr=True):
        self.set_model(model=model, copy_attr=copy_attr)

    def set_model(self, model , copy_attr=True):
        assert model is not None
        self.transformer_base = model
        if not copy_attr:
            return

        copy_attr = [
            'log',
            'log_dict'
        ]
        for k in copy_attr:
            a = getattr(self, k,None)
            if a:
                setattr(self.transformer_base, k,a)

        event_ = [
            'configure_optimizers',
            'configure_gradient_clipping',
            'lr_scheduler_step',
            'optimizer_step',
            'optimizer_zero_grad',
        ]
        for e in event_:
            a = getattr(self.transformer_base, e, None)
            if a is not None:
                setattr(self,e,a)


    def get_model_lr(self,model=None,lr=None):
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if model is not None:
            return [(model, lr)]
        return self.model.get_model_lr(model=None,lr=None) if model is None else [(model,self.config.task_specific_params['learning_rate'])]

    def compute_loss(self, *args, **kwargs):
        if len(args):
            kwargs.update(dict(args))
        return self.model.compute_loss(**kwargs)

    def forward(self, *args, **kwargs):
        if len(args):
            kwargs.update(dict(args))
        return self.compute_loss(**kwargs)


    def setup(self, stage: str) -> None:
        if self.backbone is not None:
            setattr(self.backbone, 'trainer', self.trainer)
            setattr(self.backbone, 'estimated_stepping_batches', self.trainer.estimated_stepping_batches)


    def get_named_parameters(self,*args,**kwargs):
        training_args = self.training_args
        model_attrs = self.get_model_lr(*args,**kwargs)
        no_decay = ["bias", "LayerNorm.weight"]
        def __get_named_parameters(a : nn.Module):
            return [
                {
                    "params": [p for n, p in a.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": training_args.weight_decay, "lr": lr,
                },
                {
                    "params": [p for n, p in a.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0, "lr": lr,
                },
            ]

        opt = []
        a: nn.Module
        for a, lr in model_attrs:
            opt += __get_named_parameters(a)
        return opt

    def configure_optimizers(self):
        return configure_optimizers(self.get_named_parameters(),
                                    self.training_args,
                                    self.trainer.estimated_stepping_batches,
                                    self.get_model_lr())


    def manual_backward(self,loss: Union[Tensor,Dict], *args: Any, **kwargs: Any):
        if isinstance(loss,dict):
            loss = loss['loss']
        return super(TransformerLightningModule, self).manual_backward(loss)



    def adv_training_step(self,batch):
        mode = self.training_args.adv['mode']
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
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
            scheduler and scheduler.step()
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
            setattr(self.get_embeddings_module().embeddings, 'forward', forward_fn)
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
            scheduler and scheduler.step()
            self.model.zero_grad()

            setattr(self.get_embeddings_module().embeddings, 'forward', self.embeddings_forward_fn)
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
            scheduler and scheduler.step()
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
            scheduler and scheduler.step()
            self.model.zero_grad()
        elif mode == 'free_local':
            if not hasattr(self.adversarial,'delta_'):
                setattr(self.adversarial,'delta_', torch.zeros((batch['input_ids'].size(0),self.config.max_position_embeddings, self.config.hidden_size),requires_grad=True).to(batch['input_ids'].device))
            delta = getattr(self.adversarial,'delta_')
            def forward_fn(*args, **kwargs):
                embedding_output = self.embeddings_forward_fn(*args, **kwargs)
                embedding_output += delta[:embedding_output.size(0),:embedding_output.size(1)]
                return embedding_output

            setattr(self.get_embeddings_module().embeddings, 'forward', forward_fn)
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
                scheduler and scheduler.step()
                self.model.zero_grad()

            setattr(self.get_embeddings_module().embeddings, 'forward', self.embeddings_forward_fn)
        elif mode == 'free':
            for _ in range(self.training_args.adv['minibatch_replays']):
                opt.zero_grad()
                loss = self.training_step_fn(batch)
                self.manual_backward(loss)
                if gradient_clip_val is not None:
                    self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
                opt.step()
                self.adversarial.attack(epsilon=epsilon)
                scheduler and scheduler.step()
                self.model.zero_grad()
        else:
            opt.zero_grad()
            loss = self.training_step_fn(batch)
            self.manual_backward(loss)
            if gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=gradient_clip_val)
            opt.step()
            scheduler and scheduler.step()
            self.model.zero_grad()
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if dataclasses.is_dataclass(outputs):
            self.log('loss', outputs.loss, prog_bar=True)
        elif isinstance(outputs, dict):
            self.log_dict(outputs, prog_bar=True)
        else:
            self.log('loss', outputs, prog_bar=True)

    def training_step(self, batch):
        if not isinstance(batch, dict):
            batch = dict(batch)
        outputs = self.compute_loss(**batch)
        if dataclasses.is_dataclass(outputs):
            return outputs.loss
        if isinstance(outputs,(tuple,list)):
            return outputs[0]
        return outputs

    def validation_step(self, batch, batch_idx, **kwargs):
        if not isinstance(batch, dict):
            batch = dict(batch)
        outputs = self.compute_loss(**batch)
        outputs = apply_to_collection(outputs,dtype=torch.Tensor, function=lambda x: x.detach().numpy())
        if isinstance(outputs, (tuple, list)):
            outputs = {
                "loss": outputs[0],
                "outputs": outputs[1:]
            }
        return outputs

    def test_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            batch = dict(batch)
        outputs = self.compute_loss(**batch)
        outputs = apply_to_collection(outputs,dtype=torch.Tensor, function=lambda x: x.detach().numpy())
        if isinstance(outputs, (tuple, list)):
            outputs = {
                "outputs": outputs
            }
        return outputs

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()