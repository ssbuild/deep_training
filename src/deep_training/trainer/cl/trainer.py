# -*- coding: utf-8 -*-
# @Time:  0:37
# @Author: tk
# @File：trainer
import dataclasses
import importlib
import json
import argparse
import math
import os
import random
import re
import resource
import shutil
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Union, Optional, Callable, List, Tuple, Dict, Any

import numpy as np
from lightning_utilities.core.apply_func import apply_to_collection
from packaging import version
from datasets import Dataset
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, PreTrainedModel, \
    PreTrainedTokenizerBase, DataCollator, get_scheduler, Adafactor, is_bitsandbytes_available, DefaultFlowCallback, \
    ProgressCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import OPTIMIZER_NAME, SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import CallbackHandler, PrinterCallback, TrainerState, TrainerControl
from transformers.trainer_pt_utils import get_parameter_names, IterableDatasetShard, reissue_pt_warnings
from transformers.trainer_utils import has_length, PREFIX_CHECKPOINT_DIR, number_of_arguments

from ...nlp.models.petl import PetlModel, PromptModel
from ...nlp.optimizer.optimizer import OptimizerNames
from transformers.utils import strtobool, logging
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from ...data_helper import TrainingArgumentsCL

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import (
    GeminiPlugin,
    LowLevelZeroPlugin,
    HybridParallelPlugin,
    TorchDDPPlugin,
)
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper



logger = logging.get_logger(__name__)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor



def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"




def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)

class TrainerCL:
    def __init__(self,
                 model: Union[ PreTrainedModel, nn.Module ] = None,
                 args: TrainingArgumentsCL = None,
                 data_collator: Optional[ DataCollator ] = None,
                 train_dataset: Optional[ Union[Dataset,DataLoader] ] = None,
                 eval_dataset: Optional[ Union[ Dataset,DataLoader, Dict[ str, Dataset ] ] ] = None,
                 tokenizer: Optional[ PreTrainedTokenizerBase ] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 callbacks: Optional[ List[ TrainerCallback ] ] = None,
                 optimizers: Tuple[ torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR ] = (None, None),
                 **kwargs):


        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        self.model = model
        self.args: Optional[TrainingArgumentsCL] = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.callbacks = callbacks
        self.optimizer, self.lr_scheduler = optimizers

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        colossalai.launch_from_torch({})
        self.coordinator = DistCoordinator()

        if self.coordinator.is_master():
            tensorboard_dir = self.args.logging_dir or self.args.output_dir
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)

        default_callbacks = DEFAULT_CALLBACKS
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        # self.callback_handler.train_dataloader = train_dataloader

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero,
            is_world_process_zero=self.is_world_process_zero,
        )
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        assert self.args.gradient_accumulation_steps == 1

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformer.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformer.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model


    def _setup_plugin(self):
        args = self.args
        mixed_precision = "fp16" if self.args.fp16 else "bf16"
        plugin_mode,plugin_args = (args.strategy,None) if isinstance(args.strategy,str) else (args.strategy.pop("name","ddp"),args.strategy)

        if plugin_args and plugin_mode != "ddp":
            plugin_args.update(dict(
                precision=mixed_precision,
                max_norm=args.max_grad_norm,
            ))

        if plugin_mode == "ddp":
            plugin_args = plugin_args or dict(
                broadcast_buffers= True,
                bucket_cap_mb = 25,
                find_unused_parameters = False,
                check_reduction = False,
                gradient_as_bucket_view = False,
                static_graph = False,
            )
            plugin = TorchDDPPlugin(**plugin_args)
        elif plugin_mode == "gemini":
            plugin_args = plugin_args or dict(
                precision=mixed_precision,
                initial_scale=2 ** 16,
                max_norm=args.max_grad_norm,
            )
            plugin = GeminiPlugin(**plugin_args)
        elif plugin_mode == "gemini_auto":
            plugin_args = plugin_args or dict(
                precision=mixed_precision,
                placement_policy="auto",
                initial_scale=2 ** 16,
                max_norm=args.max_grad_norm,
            )
            plugin = GeminiPlugin(**plugin_args)
        elif plugin_mode == "zero2":
            plugin_args =  plugin_args or dict(
                stage=2,
                precision=mixed_precision,
                initial_scale=2 ** 16,
                max_norm=args.max_grad_norm,
            )
            plugin = LowLevelZeroPlugin(**plugin_args)
        elif plugin_mode == "zero2_cpu":
            plugin_args = plugin_args or dict(
                stage=2,
                precision=mixed_precision,
                initial_scale=2 ** 16,
                cpu_offload=True,
                max_norm=args.max_grad_norm,
            )
            plugin = LowLevelZeroPlugin(**plugin_args)
        elif plugin_mode == "3d":
            plugin_args = plugin_args or dict(
                tp_size=1,
                pp_size=1,
                zero_stage=1,
                max_norm=args.max_grad_norm,
                precision=mixed_precision,
            )
            plugin = HybridParallelPlugin(**plugin_args)
        else:
            raise ValueError(f"Unknown plugin {args.strategy}")

        return plugin

    def setup_model(self):
        plugin = self._setup_plugin()
        booster = Booster(plugin=plugin)
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        model, optimizer, _, dataloader, lr_scheduler = booster.boost(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        self.plugin = plugin
        self.booster = booster
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler






    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def get_decay_parameter_names(self, model) -> List[ str ]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [ name for name in decay_parameters if "bias" not in name ]
        return decay_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")



        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArgumentsCL) -> Tuple[ Any, Any ]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArgumentsCL`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[ key ] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }


        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAM_HYBRID_CL:
            optimizer_cls = colossalai.nn.optimizer.hybrid_adam.HybridAdam
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAM_CPU_CL:
            optimizer_cls = colossalai.nn.optimizer.cpu_adam.CPUAdam
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAM_FUSED_CL:
            optimizer_cls = colossalai.nn.optimizer.fused_adam.FusedAdam
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_HF:
            from ...nlp.optimizer.optimizer import AdamWHF
            optimizer_cls = AdamWHF
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [ OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED ]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim in [
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
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamW
                elif "lion" in args.optim:
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}

                bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            if is_bitsandbytes_available() and version.parse(
                    importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
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
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[ int ] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch[ "input_ids" ].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens

    @property
    def rank(self):
        return self.coordinator.rank

    @property
    def world_size(self):
        return self.coordinator.world_size

    @property
    def is_local_process_zero(self):
        return self.coordinator.local_rank == 0

    @property
    def is_world_process_zero(self):
        return self.coordinator.is_master()

    def train(self,start_epoch=0,start_step=0, trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,):
        self._train_loop(start_epoch=start_epoch,start_step=start_step,trial=trial,ignore_keys_for_eval=ignore_keys_for_eval,**kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[ str, Union[ torch.Tensor, Any ] ]) -> Union[torch.Tensor,Dict,Any]:
        device = get_current_device()
        batch = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        loss = model(**batch)
        return loss


    def _train_loop(self,start_epoch=0,start_step=0,
                    trial: Union["optuna.Trial", Dict[str, Any]] = None,
                    ignore_keys_for_eval: Optional[List[str]] = None,
                    **kwargs,):
        args = self.args

        model = self.model
        train_dataloader = self.train_dataset
        coordinator = self.coordinator

        writer = self.writer


        total_train_batch_size = self.args.per_device_train_batch_size * args.gradient_accumulation_steps * self.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        model_numel = get_model_numel(model)
        # coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {format_numel_str(model_numel)}")

        if self.optimizer is None or self.lr_scheduler is None:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.setup_model()

        if not isinstance(self.optimizer, OptimizerWrapper):
            self.optimizer = OptimizerWrapper(self.optimizer)

        booster = self.booster
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        model = self.model

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero
        self.state.is_world_process_zero = self.is_world_process_zero

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_skipped = 0
        total_batched_samples = 0
        for epoch in range(start_epoch, num_train_epochs):
            # train_dataloader.sampler.set_epoch(epoch=epoch)
            num_steps_per_epoch = len(train_dataloader)

            steps_in_epoch = (
                len(train_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            step = -1
            with tqdm(
                    iterable=enumerate(train_dataloader, start=start_step),
                    desc=f"Epoch {epoch}",
                    disable=not coordinator.is_master(),
                    total=num_steps_per_epoch,
                    initial=start_step,
            ) as pbar:
                for step, batch in pbar:
                    total_batched_samples += 1
                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    loss_obj = self.training_step(model, batch)

                    if dataclasses.is_dataclass(loss_obj):
                        loss_obj = loss_obj.loss
                    elif isinstance(loss_obj, (list, tuple)):
                        loss_obj = loss_obj[0]

                    if isinstance(loss_obj, dict):
                        loss = loss_obj["loss"]
                    else:
                        loss = loss_obj

                    booster.backward(loss=loss, optimizer=optimizer)

                    all_reduce_mean(tensor=loss)

                    loss_obj = apply_to_collection(loss_obj, dtype=torch.Tensor, function=lambda x: x.detach())
                    loss = loss.detach()

                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    if coordinator.is_master():
                        global_step = epoch * num_steps_per_epoch + step
                        if isinstance(loss_obj,dict):
                            for k,v in loss_obj.items():
                                writer.add_scalar(tag=k, scalar_value=v.item(), global_step=global_step)
                        else:
                            writer.add_scalar(tag="Loss", scalar_value=loss.item(), global_step=global_step)
                        writer.add_scalar(
                            tag="Learning Rate",
                            scalar_value=lr_scheduler.get_last_lr()[ 0 ],
                            global_step=global_step,
                        )
                    is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                    ):
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(loss, model, trial, epoch, step , ignore_keys_for_eval)

                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                    # Delete CUDA cache.
                    # del batch, batch_labels, batch_output, loss
                    torch.cuda.empty_cache()
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True


                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(loss, model, trial, epoch,step, ignore_keys_for_eval)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    def _get_output_dir(self, trial):
        run_dir = self.args.output_dir
        return run_dir



    def _save_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
            epoch: int,
            step: int,
            batch_size: int,
            coordinator: DistCoordinator,
            trial = None,
    ) -> None:
        """
        Save model checkpoint, optimizer, LR scheduler and intermedidate running states.
        """
        booster = self.booster
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        os.makedirs(output_dir, exist_ok=True)

        if isinstance(model, ModelWrapper):
            model_unwrap = model.unwrap()
        else:
            model_unwrap = model

        if isinstance(model_unwrap.backbone,(PeftModel,PetlModel,PromptModel)):
            if coordinator.is_master():
                model_unwrap.backbone.save_pretrained(output_dir)
        else:
            booster.save_model(model, output_dir, shard=True,use_safetensors=self.args.save_safetensors)

        try:
            booster.save_optimizer(optimizer, os.path.join(output_dir, "optimizer"), shard=True)
        except:
            ...
        booster.save_lr_scheduler(lr_scheduler, os.path.join(output_dir, "lr_scheduler"))
        running_states = {
            "epoch": epoch,
            "step": step,
            "sample_start_index": step * batch_size,
        }
        if coordinator.is_master():
            save_json(running_states, os.path.join(output_dir, "running_states.json"))

    @classmethod
    def _load_checkpoint(
            cls,
            load_dir: Union[str, os.PathLike],
            booster: Booster,
            model: torch.nn.Module,
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
    ) -> Tuple[int, int, int]:
        """
        Load model checkpoint, optimizer, LR scheduler and intermedidate running states.
        """
        # Update booster params states.
        booster.load_model(model=model, checkpoint=load_dir)
        try:
            booster.load_optimizer(optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer"))
            booster.load_lr_scheduler(lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler"))
        except:
            ...

        running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
        return (
            running_states["epoch"],
            running_states["step"],
            running_states["sample_start_index"],
        )
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch,step, ignore_keys_for_eval):
        # if self.control.should_log:
        #     if is_torch_tpu_available():
        #         xm.mark_step()
        #
        #     logs: Dict[ str, float ] = {}
        #
        #     # all_gather + mean() to get average loss over all processes
        #     tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
        #
        #     # reset tr_loss to zero
        #     tr_loss -= tr_loss
        #
        #     logs[ "loss" ] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        #     logs[ "learning_rate" ] = self._get_learning_rate()
        #
        #     self._total_loss_scalar += tr_loss_scalar
        #     self._globalstep_last_logged = self.state.global_step
        #     self.store_flos()
        #
        #     self.log(logs)

        metrics = None
        # if self.control.should_evaluate:
        #     if isinstance(self.eval_dataset, dict):
        #         metrics = {}
        #         for eval_dataset_name, eval_dataset in self.eval_dataset.items():
        #             dataset_metrics = self.evaluate(
        #                 eval_dataset=eval_dataset,
        #                 ignore_keys=ignore_keys_for_eval,
        #                 metric_key_prefix=f"eval_{eval_dataset_name}",
        #             )
        #             metrics.update(dataset_metrics)
        #     else:
        #         metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        #     self._report_to_hp_search(trial, self.state.global_step, metrics)
        #
        #     # Run delayed LR scheduler now that metrics are populated
        #     if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         metric_to_check = self.args.metric_for_best_model
        #         if not metric_to_check.startswith("eval_"):
        #             metric_to_check = f"eval_{metric_to_check}"
        #         self.lr_scheduler.step(metrics[ metric_to_check ])

        if self.control.should_save:
            self.coordinator.print_on_master("\nStart saving model checkpoint with running states")
            self._save_checkpoint(
                model=model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=epoch,
                step=step + 1,
                batch_size=self.args.per_device_train_batch_size,
                coordinator=self.coordinator,
                trial=trial,
            )
            self.coordinator.print_on_master(
                f"Saved checkpoint at epoch {epoch} step {step + 1} at folder {self.args.output_dir}"
            )

            # self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            run_dir = self._get_output_dir(trial=trial)
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)
