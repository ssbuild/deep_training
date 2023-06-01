# -*- coding: utf-8 -*-
# @Time:  22:06
# @Author: tk
# @File：fabric

import copy
import logging
import os
import re
import warnings
from collections import OrderedDict
from typing import Optional, Any, Dict
import lightning as pl
import torch
from lightning.fabric.strategies import DeepSpeedStrategy as DeepSpeedStrategyFabric
from torch import Tensor
from .utils import gather_ds_state_dict

__all__ = [
    'FabricModelCheckpoint'
]


class FabricModelCheckpoint:
    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"

    def __init__(self,
                 dirpath:  str,
                 filename: Optional[str] = None,
                 rank=0,# 执行节点
                 every_n_train_steps: Optional[int] = None,
                 every_n_epochs: Optional[int] = None,
                 skip_n_train_steps :  Optional[int] = None,
                 skip_n_epochs: Optional[int] = None,
                 monitor=None,
                 mode='min',
                 save_last: Optional[bool] = None,
                 save_weights_only=False,
                 save_top_k = 1,
                 **kwargs):

        self.__every_n_train_steps = every_n_train_steps
        self.__every_n_epochs = every_n_epochs
        assert not (self.__every_n_epochs is None and self.__every_n_train_steps is None),ValueError('must set value one of [every_n_train_steps,every_n_epochs]')
        self.best = {}
        self.monitor = monitor
        self.mode = mode # min max

        self.save_weights_only = save_weights_only
        self.rank = rank

        self.last_eval_step = -1

        self.skip_n_train_steps = skip_n_train_steps
        self.skip_n_epochs = skip_n_epochs


        self._external_kwargs = kwargs

        self.lora_args = self._external_kwargs.get('lora_args', None)
        self.prompt_args = self._external_kwargs.get('prompt_args', None)

        self.dirpath = dirpath
        self.filename = filename
        self.save_last = save_last
        if self.lora_args or self.prompt_args:
            self.CHECKPOINT_NAME_LAST = "last"
            self.FILE_EXTENSION = ""

    @property
    def external_kwargs(self):
        return self._external_kwargs

    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, Tensor]:
        monitor_candidates = copy.deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    def on_get_metric( self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        return {}

    def update_best(self,val):
        flag = False
        if isinstance(val, torch.Tensor):
            if self.monitor not in self.best:
                flag = True
                self.best[self.monitor] = val
            else:
                monitor_op = torch.le if self.mode.lower() == 'min' else torch.ge
                if monitor_op(val, self.best[self.monitor]).bool().cpu().item():
                    flag = True
        else:
            warnings.warn('monitor {} is not tensor'.format(self.monitor))

        if flag:
            self.best[self.monitor] = val
        return flag

    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        monitor_candidates = self._monitor_candidates(trainer)
        monitor_candidates.update(self.on_get_metric(trainer,pl_module))
        val = monitor_candidates.get(self.monitor,None)

        is_save = False
        if val is not None:
            flag = self.update_best(val)
            if flag:
                logging.info('epoch {} ,step {} , save saving {}\n'.format(monitor_candidates['epoch'],
                                                                           monitor_candidates['step'],
                                                                           self.best[self.monitor]))
                is_save = True

        else:
            warnings.warn('monitor {} is not in metirc , save lastest checkpoint!'.format(self.monitor))

            logging.info('epoch {} ,step {} , saving\n'.format(monitor_candidates['epoch'],
                                                                monitor_candidates['step']))
            is_save = True

        if is_save:
            if self.filename is None:
                filename = 'epoch{}{}step{}{}'.format(monitor_candidates['epoch'],
                                                      self.CHECKPOINT_JOIN_CHAR,
                                                      monitor_candidates['step'],
                                                      self.FILE_EXTENSION)

            else:
                filename = '{}{}epoch{}{}step{}{}'.format(self.filename,
                                                          self.CHECKPOINT_JOIN_CHAR,
                                                          monitor_candidates['epoch'],
                                                          self.CHECKPOINT_JOIN_CHAR,
                                                          monitor_candidates['step'],
                                                          self.FILE_EXTENSION)

            self._save_checkpoint(trainer,os.path.join(self.dirpath,filename),pl_module)
            if self.save_last:
                filename = '{}{}'.format(self.CHECKPOINT_NAME_LAST,self.FILE_EXTENSION)
                self._save_checkpoint(trainer, os.path.join(self.dirpath, filename), pl_module)




    def __on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        # if trainer.world_size > 1:
        #     if self.rank >= 0 and trainer.global_rank != self.rank:
        #         return

        self.on_save_model(trainer,pl_module)


    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any,
            batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """

        flag = False
        if self.__every_n_train_steps is not None and self.__every_n_train_steps > 0:
            if trainer.global_step !=0 and trainer.global_step  % self.__every_n_train_steps == 0:
                # 由于梯度积累，已经执行过，跳过
                if self.last_eval_step != trainer.global_step:
                    if self.skip_n_train_steps is not None and trainer.global_step < self.skip_n_train_steps:
                        flag = False
                    else:
                        flag = True
        if flag:
            self.last_eval_step = trainer.global_step
            self.__on_save_model(trainer, pl_module)



    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

        if self.__every_n_epochs is not None and self.__every_n_epochs > 0:
            if trainer.current_epoch % self.__every_n_epochs == 0:
                if self.skip_n_epochs is not None and trainer.current_epoch < self.skip_n_epochs:
                    pass
                else:
                    self.__on_save_model(trainer, pl_module)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str, pl_module: "pl.LightningModule") -> None:
        bHandled = False

        if self.FILE_EXTENSION and isinstance(trainer.fabric.strategy, DeepSpeedStrategyFabric):
            self.FILE_EXTENSION = ""

        if self.lora_args or self.prompt_args:
            bHandled = True
            model = pl_module.module
            if isinstance(trainer.fabric.strategy, DeepSpeedStrategyFabric):
                checkpoints = model.backbone.get_all_state_dict()
                gather_ds_state_dict(checkpoints, filepath,
                                     zero_stage_3=trainer.fabric.strategy.zero_stage_3,
                                     is_global_zero=trainer.fabric.strategy.is_global_zero,
                                     config=model.backbone.config)
            else:
                if trainer.fabric.strategy.is_global_zero:
                    model.backbone.save_pretrained(filepath)

                    config_path = os.path.join(filepath, 'config.json')
                    if not os.path.exists(config_path):
                        model.backbone.config.save_pretrained(filepath)

                # trainer.fabric.strategy.barrier()
        if not bHandled:
            trainer.save_checkpoint(filepath, self.save_weights_only)
















    # def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str, pl_module: "pl.LightningModule") -> None:
    #     bHandled = False
    #     if self.lora_args or self.prompt_args:
    #         bHandled = True
    #         model = pl_module.modules
    #         checkpoints = model.backbone.get_all_state_dict()
    #         m = trainer.fabric.strategy.model.module if isinstance(trainer.fabric.strategy, DeepSpeedStrategyFabric) else model
    #         eng = trainer.fabric.strategy.model if isinstance(trainer.fabric.strategy, DeepSpeedStrategyFabric) else None
    #         for adapter_name, state in checkpoints.items():
    #             lora_or_prompt_config = state['config']
    #
    #             def state_dict_fn(module, destination, prefix, local_metadata):
    #                 return state['state_dict']
    #
    #             h = m._register_state_dict_hook(state_dict_fn)
    #             basename = os.path.basename(filepath)
    #             dirname = os.path.dirname(filepath)
    #             if adapter_name != 'default':
    #                 basename = adapter_name + '_' + basename
    #                 config_dir = os.path.join(dirname, adapter_name)
    #             else:
    #                 config_dir = dirname
    #             filepath_new = os.path.join(dirname, basename)
    #             fn_old = None
    #             get_zero_param_shapes_old = None
    #             if isinstance(trainer.fabric.strategy, DeepSpeedStrategyFabric):
    #                 fn_old = eng.zero_optimization_partition_gradients
    #                 eng.zero_optimization_partition_gradients = lambda: False
    #                 zero_optimizer_state = eng.zero_optimization() or eng.bfloat16_enabled()
    #                 if zero_optimizer_state:
    #                     results_list_new = []
    #                     results_list = eng._get_zero_param_shapes()
    #                     for results in results_list:
    #                         sub_module = OrderedDict()
    #                         for key, value in results.items():
    #                             key = re.sub(r'_forward_module\._TransformerLightningModule__backbone\.', '', key)
    #                             if self.lora_args:
    #                                 k1, k2 = key.split('.lora_')
    #                                 k2 = re.sub(re.compile('\.{}\.'.format(adapter_name)), '.', k2)
    #                                 key = k1 + '.lora_' + k2
    #                                 key = key.replace("modules_to_save.", "")
    #                             else:
    #                                 key = key.replace("modules_to_save.", "")
    #                                 key = key.replace(f".{adapter_name}", "")
    #                                 key = key.replace('prompt_encoder.embedding.weight', 'prompt_embeddings')
    #                             sub_module[key] = value
    #                         results_list_new.append(sub_module)
    #                     get_zero_param_shapes_old = eng._get_zero_param_shapes
    #                     eng._get_zero_param_shapes = lambda: results_list_new
    #             trainer.save_checkpoint(filepath_new, self.save_weights_only)
    #             if fn_old:
    #                 eng.zero_optimization_partition_gradients = fn_old
    #                 eng._get_zero_param_shapes = get_zero_param_shapes_old
    #             h.remove()
    #             lora_or_prompt_config.save_pretrained(config_dir)
    #
    #     if not bHandled:
    #         trainer.save_checkpoint(filepath, self.save_weights_only)
