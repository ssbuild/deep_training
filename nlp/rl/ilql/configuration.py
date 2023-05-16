# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 11:03


from dataclasses import field, dataclass
from typing import Optional, Dict, Any
from ..utils.configuration import RLConfigMixin


@dataclass
class ILQLConfig(RLConfigMixin):
    model_arch_type: Optional[str] = "causal"  # one of causal, prefixlm,seq2seq
    tau: float = field(default=0.7, metadata={"help": ""})
    gamma: float = field(default=0.99, metadata={"help": ""})
    cql_scale: float = field(default=0.1, metadata={"help": ""})
    awac_scale: float = field(default=1, metadata={"help": ""})
    alpha: float = field(default=0.001, metadata={"help": ""})
    beta: float = field(default=0, metadata={"help": ""})
    steps_for_target_q_sync: int = field(default=5, metadata={"help": ""})
    two_qs: bool = field(default=True, metadata={"help": ""})
    gen_kwargs: dict = field(default=None,
                             metadata={"help": "Additioanl kwargs for the generation"})
    minibatch_size: Optional[int] = field(default=None, metadata={"help": "minibatch_size"})

    def __post_init__(self):
        if self.gen_kwargs is None:
            self.gen_kwargs = dict(max_new_tokens=56, top_k=20, beta=1, temperature=1.0)
        assert self.model_arch_type is not None,ValueError('ppo args model_arch_type can not be None')
        self.model_arch_type = self.model_arch_type.lower()
        assert self.steps_for_target_q_sync >= 1



@dataclass
class ILQLArguments:
    ilql: ILQLConfig= field(default=None, metadata={"help": "ILQLConfig."})


    def save_pretrained(self, save_directory, **kwargs):
        if self.ilql is not None:
            self.ilql.save_pretrained(save_directory, **kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = ILQLConfig.from_pretrained(pretrained_model_name_or_path,**kwargs)
        return config

    @property
    def config(self) -> Optional[ILQLConfig]:
        if self.ilql is not None:
            return self.ilql
        return None


    def __post_init__(self):
        if self.ilql is not None and isinstance(self.ilql, dict):
            self.ilql = ILQLConfig.from_memory(self.ilql)

