# @Time    : 2023/3/2 20:55
# @Author  : tk
# @FileName: config.py
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

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
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
            raise ValueError(f"Can't find config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        kwargs.pop("subfolder", None)

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





@dataclass
class LoraArguments(LoraConfigMixin):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    lora_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    with_lora: bool = field(default=False, metadata={"help": "whether use lora"})
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
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        if self.inference_mode:
            self.merge_weights = True

        if self.target_modules is not None and len(self.target_modules) == 1:
            self.fan_in_fan_out = True
            self.enable_lora = [True, False, True]