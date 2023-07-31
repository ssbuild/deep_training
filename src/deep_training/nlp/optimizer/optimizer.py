# -*- coding: utf-8 -*-
# @Time:  15:23
# @Author: tk
# @File：optimizer

import typing
import torch
from torch import optim
from transformers.utils import ExplicitEnum, strtobool
from ..scheduler import WarmupCosineSchedule
from ...data_helper import TrainingArguments
from ..optimizer import lion,lamb
try:
    from transformers import AdamW as AdamWHF, Adafactor
except:
    AdamWHF = None


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAM = "adam"
    ADAMW_HF = "adamw_hf"
    ADAMW = "adamw"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    LION_8BIT = "lion_8bit"
    LION_CUSTOM = "lion"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    LAMB = "lamb"
    LAMB_FUSED_DP = 'lamb_fused_dp'
    ADAGRAD_CPU_DP = 'adagrad_cpu_dp'
    ADAM_CPU_DP = 'adam_cpu_dp'
    ADAM_FUSED_DP = 'adam_fused_dp'

def get_optimizer_cls_and_kwargs(optimizer_name,args: TrainingArguments) -> typing.Tuple[typing.Any, typing.Any]:
    """
    Returns the optimizer class and optimizer parameters based on the training arguments.

    Args:
        args (`TrainingArguments`):
            The training arguments for the training session.
    """

    # parse optimizer_name
    optim_args = {}
    if args.optimizer_args:
        for mapping in args.optimizer_args.replace(" ", "").split(","):
            key,value = mapping.split("=")
            optim_args[key] = value

    optimizer_kwargs = {"lr": args.learning_rate}

    adam_kwargs = {
        "betas":  tuple(args.optimizer_betas),
        "eps": args.adam_epsilon,
    }
    if optimizer_name == OptimizerNames.ADAFACTOR:
        optimizer_cls = Adafactor
        optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
    elif optimizer_name == OptimizerNames.ADAMW_HF:
        optimizer_cls = AdamWHF
        optimizer_kwargs.update(adam_kwargs)
    elif optimizer_name == OptimizerNames.ADAM:
        optimizer_cls = optim.Adam
        optimizer_kwargs.update(adam_kwargs)
    elif optimizer_name in [OptimizerNames.ADAMW,OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
        optimizer_cls = optim.AdamW
        optimizer_kwargs.update(adam_kwargs)
        if optimizer_name == OptimizerNames.ADAMW_TORCH_FUSED:
            optimizer_kwargs.update({"fused": True})
    elif optimizer_name == OptimizerNames.ADAMW_TORCH_XLA:
        try:
            from torch_xla.amp.syncfree import AdamW
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
    elif optimizer_name == OptimizerNames.ADAMW_APEX_FUSED:
        try:
            from apex.optimizers import FusedAdam
            optimizer_cls = FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
    elif optimizer_name in [OptimizerNames.LAMB_FUSED_DP, OptimizerNames.ADAGRAD_CPU_DP, OptimizerNames.ADAM_CPU_DP,OptimizerNames.ADAM_FUSED_DP]:
        if optimizer_name == OptimizerNames.LAMB_FUSED_DP:
            from deepspeed.ops.lamb import FusedLamb
            optimizer_cls = FusedLamb
            optimizer_kwargs.update(adam_kwargs)
        elif optimizer_name == OptimizerNames.ADAGRAD_CPU_DP:
            from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
            adam_kwargs.pop('betas',None)
            optimizer_cls = DeepSpeedCPUAdagrad
            optimizer_kwargs.update(adam_kwargs)
        elif optimizer_name == OptimizerNames.ADAM_CPU_DP:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer_cls = DeepSpeedCPUAdam
            optimizer_kwargs.update(adam_kwargs)
        elif optimizer_name == OptimizerNames.ADAM_FUSED_DP:
            from deepspeed.ops.adam import FusedAdam
            optimizer_cls = FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError('invalid optimizer_name ',optimizer_name)

    elif optimizer_name == OptimizerNames.LION_CUSTOM:
        optimizer_cls = lion.Lion
        optimizer_kwargs.update(adam_kwargs)
    elif optimizer_name == OptimizerNames.LAMB:
        optimizer_cls = lamb.Lamb
        optimizer_kwargs.update(adam_kwargs)
    elif optimizer_name in [
        OptimizerNames.ADAMW_BNB,
        OptimizerNames.ADAMW_8BIT,
        OptimizerNames.PAGED_ADAMW,
        OptimizerNames.PAGED_ADAMW_8BIT,
        OptimizerNames.LION,
        OptimizerNames.LION_8BIT,
        OptimizerNames.PAGED_LION,
        OptimizerNames.PAGED_LION_8BIT,
    ]:
        try:
            from bitsandbytes.optim import AdamW, Lion
            is_paged = False
            optim_bits = 32
            optimizer_cls = None
            additional_optim_kwargs = adam_kwargs
            if "paged" in optimizer_name:
                is_paged = True
            if "8bit" in optimizer_name:
                optim_bits = 8
            if "adam" in optimizer_name:
                optimizer_cls = AdamW
            elif "lion" in optimizer_name:
                optimizer_cls = Lion
                additional_optim_kwargs = {"betas": tuple(args.optimizer_betas)}

            bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
            optimizer_kwargs.update(additional_optim_kwargs)
            optimizer_kwargs.update(bnb_kwargs)
        except ImportError:
            raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
    elif optimizer_name == OptimizerNames.ADAMW_BNB:
        try:
            from bitsandbytes.optim import Adam8bit
            optimizer_cls = Adam8bit
            optimizer_kwargs.update(adam_kwargs)
        except ImportError:
            raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
    elif optimizer_name == OptimizerNames.ADAMW_ANYPRECISION:
        try:
            from torchdistx.optimizers import AnyPrecisionAdamW

            optimizer_cls = AnyPrecisionAdamW
            optimizer_kwargs.update(adam_kwargs)

            # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
            optimizer_kwargs.update(
                {
                    "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                    "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                    "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                    "compensation_buffer_dtype": getattr(
                        torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                    ),
                }
            )
        except ImportError:
            raise ValueError("Please install https://github.com/pytorch/torchdistx")
    elif optimizer_name == OptimizerNames.SGD:
        optimizer_cls = torch.optim.SGD
    elif optimizer_name == OptimizerNames.ADAGRAD:
        optimizer_cls = torch.optim.Adagrad
    else:
        raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {optimizer_name}")
    return optimizer_cls, optimizer_kwargs