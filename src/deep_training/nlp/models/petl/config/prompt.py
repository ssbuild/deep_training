# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
# @File：configuration.py
import enum
from dataclasses import dataclass, field
from typing import Union, Optional
from .petl import PetlConfigMixin

__all__ = [
    "TaskType",
    "PromptType",
    "PromptConfigBase",
    "PromptLearningConfig",
    "PromptTuningConfig",
    "PrefixTuningConfig",
    "AdaptionPromptConfig",
    "PromptEncoderReparameterizationType",
    "PromptTuningInit",
]

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
class PromptConfigBase(PetlConfigMixin):
    """
      inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    enable: bool = field(default=False, metadata={"help": "whether use prompt"})
    with_prompt: bool = field(default=False, metadata={"help": "whether use prompt"})

    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    prompt_type: str = field(default='prefix_tuning', metadata={"help": "one of prompt_tuning,p_tuning,prefix_tuning,adaption_prompt"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type, one of seq_cls,seq_2_seq_lm,causal_lm,token_cls"})
    target_dtype: Optional[Union[int, str]] = field(
        default=None,
        metadata={
            "help": "target_modules dtype , one of [\"64\", \"32\", \"16\", \"bf16\"]  or one of [16,32,64]"
        },
    )



@dataclass
class PromptLearningConfig(PromptConfigBase):
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




