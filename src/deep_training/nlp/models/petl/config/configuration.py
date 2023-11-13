# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py
from dataclasses import dataclass, field
from typing import Union, Optional, List
from ....layers.petl.constants import COMMON_LAYERS_PATTERN
from .petl_configuration import PetlConfig
from .lora_configuration import LoraConfig, AdaLoraConfig, IA3Config
from .loha_configuration import LoHaConfig
from .lokr_configuration import LoKrConfig


__all__ = [
    "PETL_TYPE_TO_CONFIG_MAPPING",
    "PetlArguments",
    "COMMON_LAYERS_PATTERN",
    "PetlConfig",
    "LoraConfig",
    "AdaLoraConfig",
    "IA3Config",
    "LoHaConfig",
    "LoKrConfig"
]


PETL_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
    "ia3": IA3Config,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
}

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
            if conf is not None:
                conf.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = PETL_TYPE_TO_CONFIG_MAPPING[PetlConfig.from_pretrained(pretrained_model_name_or_path, **kwargs).lora_type].from_pretrained(pretrained_model_name_or_path, **kwargs)
        assert config.enable , ValueError('lora config get bad enable ',config.enable)
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
    def config(self) -> Optional[Union[LoraConfig,AdaLoraConfig,IA3Config]]:
        if not self.enable:
            return None
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self, key)
            if conf is not None and conf.enable:
                return conf
        return None


    def __post_init__(self):
        self.enable = False
        for key in list(PETL_TYPE_TO_CONFIG_MAPPING.keys()):
            conf = getattr(self, key)
            if conf is not None and isinstance(self.lora, conf):
                conf = PETL_TYPE_TO_CONFIG_MAPPING[key].from_memory(conf)
                setattr(self,key,conf)
                self.enable = self.lora.enable | self.enable


