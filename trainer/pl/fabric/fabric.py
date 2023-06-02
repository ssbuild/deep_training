# -*- coding: utf-8 -*-
# @Time:  20:28
# @Author: tk
# @File：fabric
from typing import Any, Sequence
import lightning as L
import torch
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import nn
from torch.optim import Optimizer
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy

__all__ = [
    'FabricEx'
]
class FabricEx(L.Fabric):

    def _validate_setup(self, module: nn.Module, optimizers: Sequence[Optimizer]) -> None:
        if isinstance(module, _FabricModule):
            raise ValueError("A model should be passed only once to the `setup` method.")

        if isinstance(self.strategy, DeepSpeedStrategy) and 'optimizer' in self.strategy.config:
            pass
        else:
            if any(isinstance(opt, _FabricOptimizer) for opt in optimizers):
                raise ValueError("An optimizer should be passed only once to the `setup` method.")

        if isinstance(self._strategy, FSDPStrategy):
            raise RuntimeError(
                f"The `{type(self).__name__}` requires the model and optimizer(s) to be set up separately."
                " Create and set up the model first through `model = self.setup_model(model)`. Then create the"
                " optimizer and set it up: `optimizer = self.setup_optimizer(optimizer)`."
            )

    def setup(
        self,
        module: nn.Module,
        *optimizers: Optimizer,
        move_to_device: bool = True,
    ) -> Any:  # no specific return because the way we want our API to look does not play well with mypy
        """Set up a model and its optimizers for accelerated training.

        Args:
            module: A :class:`torch.nn.Module` to set up
            *optimizers: The optimizer(s) to set up (no optimizers is also possible)
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.

        Returns:
            The tuple containing wrapped module and the optimizers, in the same order they were passed in.
        """
        self._validate_setup(module, optimizers)
        original_module = module

        module = self._precision.convert_module(module)

        if move_to_device:
            module = self._move_model_to_device(model=module, optimizers=list(optimizers))

        if isinstance(self.strategy, DeepSpeedStrategy) and 'optimizer' in self.strategy.config:
            module, optimizers = self._strategy.setup_module_and_optimizers(  # type: ignore[assignment]
                module, [None]
            )
        else:

            # Let accelerator/plugin wrap and connect the models and optimizers
            if optimizers:
                module, optimizers = self._strategy.setup_module_and_optimizers(  # type: ignore[assignment]
                    module, list(optimizers)
                )
            else:
                module = self._strategy.setup_module(module)



        module = _FabricModule(module, self._precision, original_module=original_module)

        # Update the _DeviceDtypeModuleMixin's device parameter
        module.to(self.device if move_to_device else next(module.parameters(), torch.tensor(0)).device)

        optimizers = [_FabricOptimizer(optimizer=optimizer, strategy=self._strategy) for optimizer in optimizers]

        self._models_setup += 1

        if hasattr(original_module, "_fabric"):  # this is probably a LightningModule
            original_module._fabric = self  # type: ignore[assignment]
            original_module._fabric_optimizers = optimizers  # type: ignore[assignment]

        if optimizers:
            # join both types in a tuple for API convenience
            return (module, *optimizers)
        return module