# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, List, Literal, AnyStr
from transformers.utils import PushToHubMixin

WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"

_PRECISION_INPUT_INT = Literal[64, 32, 16]
_PRECISION_INPUT_STR = Literal["64", "32", "16", "bf16"]
_PRECISION_INPUT = Union[_PRECISION_INPUT_INT, _PRECISION_INPUT_STR]




@dataclass
class LoraConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from `transformers.utils.PushToHubMixin` which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    """

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

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

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)

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
class LoraBaseArguments(LoraConfigMixin):
    """
      inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    lora_type: str = field(default='lora', metadata={"help": "one of lora,adalora"})
    with_lora: bool = field(default=False, metadata={"help": "whether use lora"})
    target_dtype: Optional[Union[int, str]] = field(
        default=None,
        metadata={
            "help": "target_modules dtype , one of [\"64\", \"32\", \"16\", \"bf16\"]  or one of [16,32,64]"
        },
    )

@dataclass
class LoraConfig(LoraBaseArguments):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:

        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    target_dtype: Optional[Union[int,str]]= field(
        default=None,
        metadata={
            "help": "target_modules dtype , one of [\"64\", \"32\", \"16\", \"bf16\"]  or one of [16,32,64]"
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )

    def __post_init__(self):
        if self.lora_type is None:
            self.lora_type = 'lora'


@dataclass
class AdaLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        target_r (`int`): The target average rank of incremental matrix.
        init_r (`int`): The initial rank for each incremental matrix.
        tinit (`int`): The steps of initial fine-tuning warmup.
        tfinal (`int`): The step of final fine-tuning.
        deltaT (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        total_step (`int`): The total training steps that should be specified before training.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Intial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        if self.lora_type is None:
            self.lora_type = 'adalora'


LORA_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
}

@dataclass
class LoraArguments:
    lora: LoraConfig= field(default=None, metadata={"help": "LoraArguments."})
    adalora: AdaLoraConfig = field(default=None, metadata={"help": "AdaLoraArguments."})

    def save_pretrained(self, save_directory, **kwargs):
        if self.lora is not None:
            self.lora.save_pretrained(save_directory, **kwargs)
        elif self.adalora is not None:
            self.adalora.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = LORA_TYPE_TO_CONFIG_MAPPING[LoraBaseArguments.from_pretrained(pretrained_model_name_or_path,**kwargs).lora_type].from_pretrained(pretrained_model_name_or_path,**kwargs)
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
    def config(self) -> Optional[Union[LoraConfig,AdaLoraConfig]]:
        if not self.with_lora:
            return None
        if self.lora is not None and self.lora.with_lora:
            return self.lora
        elif self.adalora is not None and self.adalora.with_lora:
            return self.adalora
        return None


    def __post_init__(self):
        self.with_lora = False
        if self.lora is not None and isinstance(self.lora, dict):
            self.lora = LoraConfig.from_memory(self.lora)
            self.with_lora = self.lora.with_lora | self.with_lora


        if self.adalora is not None and isinstance(self.adalora, dict):
            self.adalora = AdaLoraConfig.from_memory(self.adalora)
            self.with_lora = self.adalora.with_lora | self.with_lora
