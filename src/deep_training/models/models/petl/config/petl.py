# -*- coding: utf-8 -*-
# @Time:  10:47
# @Author: tk
# @File：petl_config

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, List, Literal, AnyStr
from transformers.utils import PushToHubMixin
from .....utils.function import copy_dataclass
from ....layers.petl.constants import WEIGHTS_NAME,SAFETENSORS_WEIGHTS_NAME,CONFIG_NAME



@dataclass
class PetlConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from `transformers.utils.PushToHubMixin` which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    """

    # @property
    # def __dict__(self):
    #     return asdict(self)

    def to_dict(self):
        return asdict(self)

    def __deepcopy__(self, memodict={}):
        return self.__class__(**copy.deepcopy(self.to_dict()))

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            **kwargs:
                Additional keyword arguments passed along to the `transformers.utils.PushToHubMixin.push_to_hub`
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        kwargs.pop("subfolder",None)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def from_memory(cls,json_object: dict, **kwargs):
        config = cls(**kwargs)
        loaded_attributes = json_object
        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


@dataclass
class PetlConfig(PetlConfigMixin):
    """
      inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    lora_type: str = field(default='lora', metadata={"help": "one of lora,adalora,ia3,lycoris,lokr"})
    # 兼容旧版本
    with_lora: bool = field(default=False, metadata={"help": "whether use lora"})
    enable: bool = field(default=False, metadata={"help": "whether use lora"})
