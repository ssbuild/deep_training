# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/9 10:31
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as TrainingArgumentsHF_
from .base_args import _ArgumentsBase


@dataclass
class TrainingArgumentsHF(TrainingArgumentsHF_,_ArgumentsBase):

    data_backend: Optional[str] = field(
        default="record",
        metadata={
            "help": (
                "default data_backend."
            )
        },
    )
    learning_rate_for_task: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "learning_rate_for_task."
            )
        },
    )


    def __post_init__(self):
        super().__post_init__()
        if self.learning_rate_for_task is None:
            self.learning_rate_for_task = self.learning_rate
