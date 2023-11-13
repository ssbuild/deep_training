# -*- coding: utf-8 -*-
# @Time:  10:47
# @Author: tk
# @File：lora_config
from dataclasses import dataclass, field
from typing import Union, Optional, List
from .petl import PetlConfig


@dataclass
class LoraConfig(PetlConfig):
    """

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

    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )

    def __post_init__(self):
        if self.lora_type is None:
            self.lora_type = 'lora'


@dataclass
class AdaLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~efficient.AdaLora`].

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



@dataclass
class IA3Config(PetlConfig):
    """
    This is the configuration class to store the configuration of a [`IA3Model`].

    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply (IA)^3 to.
        feedforward_modules (`Union[List[str],str]`): The names of the modules to be treated as feedforward modules
        as in the original paper.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        modules_to_save (`List[str]`):List of modules apart from (IA)^3 layers to be set as trainable
            and saved in the final checkpoint.
        init_ia3_weights (`bool`): Whether to initialize the vectors in the (IA)^3 layers, defaults to `True`.
    """

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ia3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    feedforward_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or a regex expression of module names which are feedforward"
            "For example, ['output.dense']"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_ia3_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the vectors in the (IA)^3 layers."},
    )

    def __post_init__(self):
        if self.lora_type is None:
            self.lora_type = 'ia3'