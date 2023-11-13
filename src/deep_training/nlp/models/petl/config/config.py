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



@dataclass
class PetlArguments:
    lora: Optional[LoraConfig] = field(default=None, metadata={"help": "LoraConfig."})
    adalora: Optional[AdaLoraConfig] = field(default=None, metadata={"help": "AdaLoraConfig."})
    ia3: Optional[IA3Config] = field(default=None, metadata={"help": "IA3Config."})
    loha: Optional[LoHaConfig] = field(default=None, metadata={"help": "LoHaConfig."})
    lokr: Optional[LoKrConfig] = field(default=None, metadata={"help": "LoKrConfig."})
    def save_pretrained(self, save_directory, **kwargs):
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self,key)
            if conf is not None and conf.enable:
                conf.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = PETL_TYPE_TO_CONFIG_MAPPING[PetlConfig.from_pretrained(pretrained_model_name_or_path, **kwargs).lora_type].from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.enable = config.with_lora = config.enable | config.with_lora
        assert config.enable , ValueError('petl config get bad enable ',config.enable)
        return config

    @property
    def config(self) -> Optional[PetlConfigMixin]:
        if not self.enable:
            return None
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self, key)
            if conf is not None and (conf.enable or getattr(conf,"with_lora",None)):
                return conf
        return None


    def __post_init__(self):
        self.enable,self.with_lora = False,False
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self, key)
            if conf is not None:
                if isinstance(conf, dict):
                    conf = PETL_TYPE_TO_CONFIG_MAPPING[key].from_memory(conf)
                    setattr(self,key,conf)

                conf.with_lora = conf.enable = conf.enable | conf.with_lora
                self.enable = self.with_lora = conf.enable | self.enable


@dataclass
class PromptArguments:
    prompt: Optional[PromptLearningConfig] = field(default=None, metadata={"help": "PromptLearningConfig."})
    def save_pretrained(self, save_directory, **kwargs):
        conf = self.prompt
        if conf is not None and conf.enable:
            conf.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = PROMPT_TYPE_TO_CONFIG_MAPPING[PromptLearningConfig.from_pretrained(pretrained_model_name_or_path, **kwargs).prompt_type].from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.enable = config.with_prompt = config.enable | config.with_prompt
        assert config.enable , ValueError('petl config get bad enable ',config.enable)
        return config

    @property
    def config(self) -> Optional[PromptLearningConfig]:
        if not self.enable:
            return None
        conf = self.prompt
        if conf is not None and (conf.enable or getattr(conf,"with_prompt",None)):
            return conf
        return None


    def __post_init__(self):
        self.enable,self.with_prompt = False,False
        if self.prompt is not None:
            if isinstance(self.prompt, dict):
                self.prompt = PROMPT_TYPE_TO_CONFIG_MAPPING[self.prompt["prompt_type"]].from_memory(self.prompt)
            self.prompt.with_prompt = self.prompt.enable = self.prompt.enable | self.prompt.with_prompt
            self.enable = self.with_prompt = self.prompt.enable | self.enable

