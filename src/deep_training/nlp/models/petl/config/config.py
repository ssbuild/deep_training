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
    "PromptTuningConfig",
    "PrefixTuningConfig",
    "AdaptionPromptConfig",
    "PromptEncoderReparameterizationType",
    "PromptTuningInit",
    "PromptArguments",
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

__CONFIG_MAP__ = copy.deepcopy(PROMPT_TYPE_TO_CONFIG_MAPPING)
__CONFIG_MAP__.update(copy.deepcopy(PETL_TYPE_TO_CONFIG_MAPPING))
@dataclass
class PetlArguments:

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
            config = __CONFIG_MAP__[config_.prompt_type].from_pretrained(
                pretrained_model_name_or_path, **kwargs)
        else:
            config = __CONFIG_MAP__[config_.lora_type].from_pretrained(
                pretrained_model_name_or_path, **kwargs)
        if hasattr(config,"with_prompt"):
            config.enable  |= config.with_prompt
        else:
            config.enable |= config.with_lora

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
        for key in list(__CONFIG_MAP__.keys()):
            conf = getattr(self, key)
            if conf is not None and (conf.enable or (getattr(conf,"with_lora",None) or getattr(conf,"with_prompt",None))):
                return conf
        return None


    def __post_init__(self):
        self.enable = False
        self.with_lora = False
        self.with_prompt = False
        for key in list(__CONFIG_MAP__.keys()):
            conf = getattr(self, key)
            if conf is not None and isinstance(conf, dict):
                conf = __CONFIG_MAP__[key].from_memory(conf)
                setattr(self,key,conf)
                if hasattr(conf,"with_prompt"):
                    conf.enable |= self.with_prompt
                else:
                    conf.enable |= self.with_lora
                self.enable = conf.enable | self.enable
                self.with_lora,self.with_prompt  = conf.enable


#兼容 <= 0.2.7
PromptArguments = PetlArguments