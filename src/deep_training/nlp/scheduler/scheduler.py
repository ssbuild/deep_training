# -*- coding: utf-8 -*-
# @Time:  15:23
# @Author: tk
# @File：scheduler
import math
from typing import Union, Optional
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_constant_schedule, get_constant_schedule_with_warmup, get_inverse_sqrt_schedule

from transformers.utils import ExplicitEnum
from torch.optim import Optimizer

try:
    from transformers.optimization import get_reduce_on_plateau_schedule
except:
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    def get_reduce_on_plateau_schedule(optimizer: Optimizer):
        """
        Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.

        Return:
            `torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.
        """

        return ReduceLROnPlateau(optimizer)


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
}
def get_warmup_steps(num_training_steps: int,warmup_steps: int,warmup_ratio):
    """
    Get number of steps used for a linear warmup.
    """
    warmup_steps = (
        warmup_steps if warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)
    )
    return warmup_steps


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT or name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)