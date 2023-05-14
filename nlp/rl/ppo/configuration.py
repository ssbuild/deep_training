# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 11:03


from dataclasses import field, dataclass
from typing import Optional, Dict, Any
from ..utils.configuration import RLConfigMixin




@dataclass
class PPOConfig(RLConfigMixin):
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

