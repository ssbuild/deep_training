# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 11:03
import json
import os
import warnings
from dataclasses import field, dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np
from transformers.utils import flatten_dict, PushToHubMixin

CONFIG_NAME = 'ppo_config.json'

@dataclass
class PPOConfigMixin(PushToHubMixin):
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
class PPOConfig(PPOConfigMixin):
    model_arch_type: Optional[str] = "causal"  # one of causal, prefixlm,seq2seq
    ppo_epochs: int = field(default=4, metadata={"help": "Number of updates per batch"})
    num_rollouts: int = field(default=128, metadata={"help": "Number  of experiences to observe before learning"})
    chunk_size: int = field(default=128, metadata={"help": "Number of chunk_size of generate"})
    init_kl_coef: float = field(default=0.001, metadata={"help": "Initial value for KL coefficient"})
    target: Optional[float] = field(default=None, metadata={"help": "Target value for KL coefficient"})
    horizon: int = field(default=10000, metadata={"help": "Number of steps for KL coefficient to reach target"})
    gamma: float = field(default=1., metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    cliprange: float = field(default=0.2, metadata={"help": "Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)"})
    cliprange_value: float = field(default=0.2, metadata={"help": "Clipping range for predicted values"
                            "(observed values - cliprange_value, observed values + cliprange_value)"})
    vf_coef: float = field(default=1., metadata={"help": "Value loss scale w.r.t policy loss"})
    scale_reward: Optional[str] = field(default="ignored", metadata={"help": ""})
    ref_mean: Optional[float] = field(default=None, metadata={"help": "Number of updates per batch"})
    ref_std: Optional[float] = field(default=None, metadata={"help": "Number of updates per batch"})
    cliprange_reward: int = field(default=10, metadata={"help": "Additioanl kwargs for the generation"})
    gen_kwargs: dict = field(default=None,
                             metadata={"help": "Additioanl kwargs for the generation"})
    gen_experience_kwargs: Optional[dict] = field(default=None, metadata={"help": "Additioanl kwargs for the gen_experience_kwargs"})

    minibatch_size: Optional[int] =  field(default=None, metadata={"help": "minibatch_size"})

    def __post_init__(self):
        if self.gen_kwargs is None:
            self.gen_kwargs = dict(
            max_new_tokens=40,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )
        assert self.model_arch_type is not None,ValueError('ppo args model_arch_type can not be None')
        self.model_arch_type = self.model_arch_type.lower()



@dataclass
class PPOArguments:
    ppo: PPOConfig= field(default=None, metadata={"help": "PPOConfig."})


    def save_pretrained(self, save_directory, **kwargs):
        if self.ppo is not None:
            self.ppo.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = PPOConfig.from_pretrained(pretrained_model_name_or_path,**kwargs)
        return config

    @property
    def config(self) -> Optional[PPOConfig]:
        if self.ppo is not None:
            return self.ppo
        return None


    def __post_init__(self):
        if self.ppo is not None and isinstance(self.ppo, dict):
            self.ppo = PPOConfig.from_memory(self.ppo)

