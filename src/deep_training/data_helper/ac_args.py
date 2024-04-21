# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/9 10:31
from dataclasses import dataclass, field
from typing import Optional
from .hf_args import TrainingArgumentsHF
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


TrainingArgumentsAC =TrainingArgumentsHF