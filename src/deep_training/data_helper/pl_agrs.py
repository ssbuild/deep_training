# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/9 10:29
import os
from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers.utils import logging
from .base_args import _ArgumentsBase

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class TrainingArguments(_ArgumentsBase):

    optimizer: str = field(
        default='adamw',
        metadata={"help": "one of lamb,adam,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,"
                          "adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,"
                          "paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,"
                          "lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp"},
    )
    optimizer_args: Optional[str] = field(default=None,metadata={"help": "sample a=100,b=10 "})
    scheduler_type: str = field(
        default='linear',
        metadata={"help": "one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, "
                          "cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]"},
    )

    scheduler: dict = field(
        default=None,
        # {
        #     # StepLR
        #     "decay_rate": 0.999,
        #     "decay_steps": 100,
        # }

        # {
        #     # CosineAnnealingWarmRestarts
        #     "T_mult": 1,
        #     "rewarm_epoch_num": 2,
        # }
        metadata={"help": "StepLR:  { 'decay_rate': 0.999,'decay_steps': 100,'verbose': True} ,\
                          CAWR {'T_mult': 1, 'rewarm_epoch_num': 2,'verbose': True} ,\
                          CAL: {'rewarm_epoch_num': 2,'verbose': True} \
                          "},
    )
    adv: dict = field(
        # default_factory= lambda: {
        #     'mode': None, # None, fgm, fgsm_local, fgsm, pgd, free_local, free
        #     'emb_name=': 'embedding',
        #     'attack_iters': 2, # pgd
        #     'minibatch_replays': 2, # free
        #     'alpha': 0.1, # pgd
        #     'epsilon': 1.0 # pgd,fgm
        # },
        default=None,
        metadata={"help": "对抗训练"},
    )
    hierarchical_position: float = field(
        default=None,
        metadata={"help": "层次分解位置编码，让transformer可以处理超长文本 , 绝对位置编码有效 , None禁用 , 0 - 1 启用 "},
    )

    learning_rate : float = field(
        default=5e-5,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    learning_rate_for_task: float = field(
        default=None,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    max_epochs: int = field(
        default=-1,
        metadata={"help": "模型训练的轮数"},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "max_steps"},
    )
    optimizer_betas : tuple = field (
        default=(0.9, 0.999),
        metadata={"help": "优化器的betas值"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam优化器的epsilon值"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "gradient_accumulation_steps"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "max_grad_norm"},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "weight_decay"},
    )

    warmup_steps: float = field(
        default=0,
        metadata={"help": "warmup_steps"},
    )

    train_batch_size: int = field(
        default=16,
        metadata={"help": "train_batch_size"},
    )

    eval_batch_size: int = field(
        default=1,
        metadata={"help": "eval_batch_size"},
    )

    test_batch_size: int = field(
        default=1,
        metadata={"help": "test_batch_size"},
    )
    seed: Optional[float] = field(
        default=42,
        metadata={"help": "seed"},
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    def __post_init__(self):
        if self.learning_rate_for_task is None:
            self.learning_rate_for_task = self.learning_rate

        if self.seed is not None:
            from lightning_fabric.utilities.seed import seed_everything
            seed_everything(int(self.seed))


        assert self.hierarchical_position is None or (self.hierarchical_position >0 and self.hierarchical_position <1)

