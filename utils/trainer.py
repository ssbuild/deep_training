# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 11:38
import copy
import logging
import warnings
from typing import Optional, Any, Dict
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint,Checkpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Trainer
from torch import Tensor
__all__ = [
    'SimpleModelCheckpoint'
]

class SimpleModelCheckpoint(Checkpoint):
    def __init__(self,
                 rank=0,# 执行节点
                 every_n_train_steps: Optional[int] = None,
                 every_n_epochs: Optional[int] = None,
                 skip_n_train_steps :  Optional[int] = None,
                 skip_n_epochs: Optional[int] = None,
                 monitor=None,
                 mode='min',
                 weight_file='./best.pt',#保存权重名字
                 last_weight_file='./last.pt',#每评估一次保存一次权重
                 **kwargs):

        self.__every_n_train_steps = every_n_train_steps
        self.__every_n_epochs = every_n_epochs
        assert not (self.__every_n_epochs is None and self.__every_n_train_steps is None),ValueError('must set value one of [every_n_train_steps,every_n_epochs]')
        self.best = {}
        self.monitor = monitor
        self.mode = mode # min max

        self.rank = rank

        self.last_eval_step = -1

        self.skip_n_train_steps = skip_n_train_steps
        self.skip_n_epochs = skip_n_epochs

        self.weight_file = weight_file
        self.last_weight_file = last_weight_file
        self._external_kwargs = kwargs

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


    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        monitor_candidates = self._monitor_candidates(trainer)
        monitor_candidates.update(self.on_get_metric(trainer,pl_module))
        val = monitor_candidates.get(self.monitor,None)
        if val is not None:
            flag = False
            if isinstance(val,torch.Tensor):
                if self.monitor not in self.best:
                    self.best[self.monitor] = val
                monitor_op = torch.le if self.mode.lower() == 'min' else torch.ge
                if monitor_op(val ,self.best[self.monitor]).bool().cpu().item():
                    flag = True
            else:
                warnings.warn('monitor {} is not tensor'.format(self.monitor))
            if flag:
                self.best[self.monitor] = val
                logging.info('epoch {} ,step {} , save best {}, {}\n'.format(monitor_candidates['epoch'],
                                                                           monitor_candidates['step'],
                                                                           self.best[self.monitor],
                                                                           self.weight_file))
                trainer.save_checkpoint(self.weight_file)

            if self.last_weight_file is not None:
                logging.info('epoch {} ,step {} , save {}\n'.format(monitor_candidates['epoch'],
                                                                       monitor_candidates['step'],
                                                                       self.last_weight_file))
                trainer.save_checkpoint(self.last_weight_file)

        else:
            warnings.warn('monitor {} is not in metirc !!!'.format(self.monitor))

            logging.info('epoch {} ,step {} , save {}\n'.format(monitor_candidates['epoch'],
                                                                       monitor_candidates['step'],
                                                                       self.weight_file))
            trainer.save_checkpoint(self.weight_file)




    def __on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if trainer.world_size > 1:
            if self.rank >= 0 and trainer.global_rank != self.rank:
                return

        pl_module.eval()
        with torch.no_grad():
            self.on_save_model(trainer,pl_module)
        pl_module.train()

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
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

