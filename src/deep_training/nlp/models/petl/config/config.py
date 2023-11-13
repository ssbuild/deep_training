# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py
import copy
from dataclasses import dataclass, field
from typing import Union, Optional, List
from .prompt import *
from ....layers.petl.constants import COMMON_LAYERS_PATTERN
from .petl import PetlConfig, PetlConfigMixin
from .lora import LoraConfig, AdaLoraConfig, IA3Config
from .loha import LoHaConfig
from .lokr import LoKrConfig


__all__ = [
    "PETL_TYPE_TO_CONFIG_MAPPING",
    "PetlArguments",
    "COMMON_LAYERS_PATTERN",
    "PetlConfig",
    "LoraConfig",
    "AdaLoraConfig",
    "IA3Config",
    "LoHaConfig",
    "LoKrConfig",
    #prompt
    "TaskType",
    "PromptType",
    "PromptConfigBase",
    "PromptLearningConfig",
    "PromptLearningConfig",
    "PrefixTuningConfig",
    "AdaptionPromptConfig",
]

PROMPT_TYPE_TO_CONFIG_MAPPING = {
    "prompt_tuning": PromptTuningConfig,
    "p_tuning": PromptTuningConfig,
    "prefix_tuning": PrefixTuningConfig,
    "adaption_prompt": AdaptionPromptConfig
}


PETL_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
    "ia3": IA3Config,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
}

@dataclass
class PetlArguments:
    CONFIG_MAP = copy.deepcopy(PROMPT_TYPE_TO_CONFIG_MAPPING).update(copy.deepcopy(PETL_TYPE_TO_CONFIG_MAPPING))
    lora: LoraConfig= field(default=None, metadata={"help": "LoraConfig."})
    adalora: AdaLoraConfig = field(default=None, metadata={"help": "AdaLoraConfig."})
    ia3: IA3Config = field(default=None, metadata={"help": "IA3Config."})
    loha: LoHaConfig = field(default=None, metadata={"help": "LoHaConfig."})
    lokr: LoHaConfig = field(default=None, metadata={"help": "LoKrConfig."})
    def save_pretrained(self, save_directory, **kwargs):
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self,key)
            if conf is not None and conf.enable:
                conf.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_ = PetlConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config_,"prompt_type"):
            config = cls.CONFIG_MAP[config_.prompt_type].from_pretrained(
                pretrained_model_name_or_path, **kwargs)
        else:
            config = cls.CONFIG_MAP[config_.lora_type].from_pretrained(
                pretrained_model_name_or_path, **kwargs)

        assert config.enable , ValueError('petl config get bad enable ',config.enable)
        # config = cls()
        # config.lora = None
        # config.adalora = None
        # setattr(config,obj.lora_type,obj)
        # if config.lora is not None:
        #     config.enable = config.lora.enable
        # if config.adalora is not None:
        #     config.enable = config.adalora.enable
        return config

    @property
    def config(self) -> Optional[PetlConfigMixin]:
        if not self.enable:
            return None
        for key in list(self.CONFIG_MAP.keys()):
            conf = getattr(self, key)
            if conf is not None and (conf.enable or (getattr(conf,"with_lora",None) or getattr(conf,"with_prompt",None))):
                return conf
        return None


    def __post_init__(self):
        self.enable = False
        self.with_lora = False
        self.with_prompt = False
        for key in list(self.CONFIG_MAP.keys()):
            conf = getattr(self, key)
            if conf is not None and isinstance(conf, dict):
                conf = self.CONFIG_MAP[key].from_memory(conf)
                setattr(self,key,conf)
                self.enable = conf.enable | self.enable
                self.with_lora = getattr(conf,"with_lora",None) | self.with_lora
                self.with_prompt = getattr(conf, "with_prompt", None) | self.with_prompt

                self.with_lora = self.enable = self.enable or self.with_lora
                self.with_prompt = self.enable = self.enable or self.with_prompt


