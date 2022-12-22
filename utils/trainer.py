# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 11:38
from typing import Optional, Any, Dict
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint,Checkpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Trainer

__all__ = [
    'CheckpointCallback'
]


class CheckpointCallback(Checkpoint):
    def __init__(self,
                 rank=0,# 执行节点
                 every_n_train_steps: Optional[int] = None,
                 every_n_epochs: Optional[int] = 1,
                 monitor='val_f1'):

        self.__every_n_train_steps = every_n_train_steps
        self.__every_n_epochs = every_n_epochs
        assert not (self.__every_n_epochs is None and self.__every_n_train_steps is None)
        self.best = {}
        self.monitor = monitor
        self.rank = rank


    def on_save_model(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        ...

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

        if self.__every_n_train_steps is not None and self.__every_n_train_steps > 0:
            if trainer.global_step % self.__every_n_train_steps == 0:
                self.__on_save_model(trainer,pl_module)



    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """


        if self.__every_n_epochs is not None and self.__every_n_epochs > 0:
            if trainer.current_epoch % self.__every_n_epochs == 0:
                self.__on_save_model(trainer, pl_module)

