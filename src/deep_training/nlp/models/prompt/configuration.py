# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py
import enum
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

class TaskType(str, enum.Enum):
    SEQ_CLS = "seq_cls"
    SEQ_2_SEQ_LM = "seq_2_seq_lm"
    CAUSAL_LM = "causal_lm"
    TOKEN_CLS = "token_cls"

class PromptType(str, enum.Enum):
    PROMPT_TUNING = "prompt_tuning"
    P_TUNING = "p_tuning"
    PREFIX_TUNING = "prefix_tuning"
    ADAPTION_PROMPT = "adaption_prompt"

@dataclass
class PromptConfigMixin(PushToHubMixin):
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

    @classmethod
    def from_memory(cls,json_object: dict, **kwargs):
        config = cls(**kwargs)
        loaded_attributes = json_object
        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


@dataclass
class PromptBaseArguments(PromptConfigMixin):
    """
      inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    prompt_type: str = field(default='prefix_tuning', metadata={"help": "one of prompt_tuning,p_tuning,prefix_tuning,adaption_prompt"})
    with_prompt: bool = field(default=False, metadata={"help": "whether use lora"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type, one of seq_cls,seq_2_seq_lm,causal_lm,token_cls"})
    target_dtype: Optional[Union[int, str]] = field(
        default=None,
        metadata={
            "help": "target_modules dtype , one of [\"64\", \"32\", \"16\", \"bf16\"]  or one of [16,32,64]"
        },
    )



@dataclass
class PromptLearningConfig(PromptBaseArguments):
    """
    This is the base configuration class to store the configuration of [`PrefixTuning`], [`PromptEncoder`], or
    [`PromptTuning`].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})



@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.prompt_type = PromptType.PREFIX_TUNING



class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"

@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEncoder`].

    Args:
        encoder_reparameterization_type (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The type of reparameterization to use.
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        encoder_num_layers (`int`): The number of layers of the prompt encoder.
        encoder_dropout (`float`): The dropout probability of the prompt encoder.
    """

    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )

    def __post_init__(self):
        self.prompt_type = PromptType.P_TUNING


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"

@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    def __post_init__(self):
        self.prompt_type = PromptType.PROMPT_TUNING
@dataclass
class AdaptionPromptConfig(PromptLearningConfig):
    """Stores the configuration of an [`AdaptionPromptModel`]."""

    target_modules: str = field(
        default=None, metadata={"help": "Name of the attention submodules to insert adaption prompts into."}
    )
    adapter_len: int = field(default=None, metadata={"help": "Number of adapter tokens to insert"})
    adapter_layers: int = field(default=None, metadata={"help": "Number of adapter layers (from the top)"})

    def __post_init__(self):
        self.prompt_type = PromptType.ADAPTION_PROMPT





PROMPT_TYPE_TO_CONFIG_MAPPING = {
    "prompt_tuning": PromptTuningConfig,
    "p_tuning": PromptTuningConfig,
    "prefix_tuning": PrefixTuningConfig,
    "adaption_prompt": AdaptionPromptConfig
}



@dataclass
class PromptArguments:
    prompt: PromptLearningConfig= field(default=None, metadata={"help": "prompt."})

    def save_pretrained(self, save_directory, **kwargs):
        if self.prompt is not None:
            self.prompt.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = PROMPT_TYPE_TO_CONFIG_MAPPING[PromptBaseArguments.from_pretrained(pretrained_model_name_or_path,**kwargs).prompt_type].from_pretrained(pretrained_model_name_or_path,**kwargs)
        assert config.with_prompt , ValueError('prompt config get bad with_prompt ',config.with_prompt)
        return config

    @property
    def config(self) -> Optional[PromptLearningConfig]:
        if not self.with_prompt:
            return None
        if self.prompt is not None and self.prompt.with_prompt:
            return self.prompt
        return None


    def __post_init__(self):
        self.with_prompt = False
        if self.prompt is not None and isinstance(self.prompt, dict):
            prompt_type = self.prompt.get('prompt_type','prefix_tuning')
            self.prompt = PROMPT_TYPE_TO_CONFIG_MAPPING[prompt_type].from_memory(self.prompt)
            self.with_prompt = self.prompt.with_prompt | self.with_prompt

