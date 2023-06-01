# -*- coding: utf-8 -*-
# @Time:  20:21
# @Author: tk
# @File：deepspeed.py

from typing import List, Tuple
from lightning.fabric.strategies import DeepSpeedStrategy
from torch.nn import Module
from torch.optim import Optimizer

__all__ = [
    'DeepSpeedStrategyEx'
]
class DeepSpeedStrategyEx(DeepSpeedStrategy):

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple["deepspeed.DeepSpeedEngine", List[Optimizer]]:
        """Set up a model and multiple optimizers together.

        Currently, only a single optimizer is supported.

        Return:
            The model wrapped into a :class:`deepspeed.DeepSpeedEngine` and a list with a single
            deepspeed optimizer.
        """
        if len(optimizers) != 1:
            raise ValueError(
                f"Currently only one optimizer is supported with DeepSpeed."
                f" Got {len(optimizers)} optimizers instead."
            )

        self._deepspeed_engine, optimizer = self._initialize_engine(module, optimizers[0])
        self._set_deepspeed_activation_checkpointing()
        return self._deepspeed_engine, [optimizer]