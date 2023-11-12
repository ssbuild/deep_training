# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py
from dataclasses import dataclass, field
from typing import Union, Optional, List
from ..config.lora_configuration import LoraConfig, AdaLoraConfig, IA3Config
from ..config.petl_configuration import PetlConfig

COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks", "layer"]


LORA_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
    "ia3": IA3Config
}

@dataclass
class PetlArguments:
    lora: LoraConfig= field(default=None, metadata={"help": "LoraConfig."})
    adalora: AdaLoraConfig = field(default=None, metadata={"help": "AdaLoraConfig."})
    ia3: IA3Config = field(default=None, metadata={"help": "IA3Config."})

    def save_pretrained(self, save_directory, **kwargs):
        if self.lora is not None:
            self.lora.save_pretrained(save_directory, **kwargs)
        elif self.adalora is not None:
            self.adalora.save_pretrained(save_directory, **kwargs)
        elif self.ia3 is not None:
            self.ia3.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = LORA_TYPE_TO_CONFIG_MAPPING[PetlConfig.from_pretrained(pretrained_model_name_or_path, **kwargs).lora_type].from_pretrained(pretrained_model_name_or_path, **kwargs)
        assert config.with_lora , ValueError('lora config get bad with_lora ',config.with_lora)
        # config = cls()
        # config.lora = None
        # config.adalora = None
        # setattr(config,obj.lora_type,obj)
        # if config.lora is not None:
        #     config.with_lora = config.lora.with_lora
        # if config.adalora is not None:
        #     config.with_lora = config.adalora.with_lora
        return config

    @property
    def config(self) -> Optional[Union[LoraConfig,AdaLoraConfig,IA3Config]]:
        if not self.with_lora:
            return None
        if self.lora is not None and self.lora.with_lora:
            return self.lora
        elif self.adalora is not None and self.adalora.with_lora:
            return self.adalora
        elif self.ia3 is not None and self.ia3.with_lora:
            return self.ia3
        return None


    def __post_init__(self):
        self.with_lora = False
        if self.lora is not None and isinstance(self.lora, dict):
            self.lora = LoraConfig.from_memory(self.lora)
            self.with_lora = self.lora.with_lora | self.with_lora


        if self.adalora is not None and isinstance(self.adalora, dict):
            self.adalora = AdaLoraConfig.from_memory(self.adalora)
            self.with_lora = self.adalora.with_lora | self.with_lora

        if self.ia3 is not None and isinstance(self.ia3, dict):
            self.ia3 = IA3Config.from_memory(self.ia3)
            self.with_lora = self.ia3.with_lora | self.with_lora

