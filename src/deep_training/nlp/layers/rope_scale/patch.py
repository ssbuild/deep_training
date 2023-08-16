from torch import nn
from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = [
    'RotaryDynamicScaledArguments',
    'RotaryDynamicPartNtkArguments',
    'RotaryNtkScaledArguments',
    'RotaryLinearScaledArguments',
    'RotaryPartNtkScaledArguments',
    "patch_for_dynamic_scaled_rotary_embeddings",
    "patch_for_dynamic_part_ntk_rotary_embeddings",
    "patch_for_ntk_scaled_rotary_embeddings",
    "patch_for_linear_scaled_rotary_embeddings",
    "patch_for_part_ntk_scaled_rotary_embeddings",
    "inject_rope_scale_layer"
]

def patch_for_dynamic_scaled_rotary_embeddings(model,name='rotary_emb',max_position_embeddings=None,
                                               base=10000, ntk=False):
    assert name
    from .DynamicScaledRotary import DynamicScaledRotary
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model,n,DynamicScaledRotary(dim=dim,
                                                max_position_embeddings= max_position_embeddings or p.max_position_embeddings,
                                                ntk=ntk,
                                                base=base,
                                                device=inv_freq.device))

def patch_for_dynamic_part_ntk_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=2048,original_max_position_embeddings=None ,
                                                 base=10000, ntk_factor=1.0, extrapolation_factor=1.0, finetuned=False):
    assert name
    from .DynamicPartNTKScaledRotary import DynamicPartNTKScaledRotary
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model,n,DynamicPartNTKScaledRotary(dim=dim,
                                                       max_position_embeddings=max_position_embeddings,
                                                       original_max_position_embeddings=original_max_position_embeddings or p.max_position_embeddings,
                                                       base=base,
                                                       ntk_factor=ntk_factor,
                                                       extrapolation_factor=extrapolation_factor,
                                                       finetuned=finetuned,
                                                       device=inv_freq.device))

def patch_for_ntk_scaled_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=None, base=10000, alpha=1.0):
    assert name
    from .NTKScaledRotary import NTKScaledRotary
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model,n,NTKScaledRotary(dim=dim,
                                            max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                                            base=base,
                                            alpha=alpha,
                                            device=inv_freq.device))

def patch_for_linear_scaled_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=None, base=10000, scale=1.0):
    assert name
    from .LinearScaledRotary import LinearScaledRotary
    for n, p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p, 'inv_freq', None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model, n, LinearScaledRotary(dim=dim,
                                              max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                                              base=base,
                                              scale=scale,
                                              device=inv_freq.device))

def patch_for_part_ntk_scaled_rotary_embeddings(model,name='rotary_emb', original_max_position_embeddings=None,max_position_embeddings=2048,
                                                base=10000, scale=1.0, ntk_factor=1.0, extrapolation_factor=1.0):
    assert name
    from .PartNTKScaledRotary import PartNTKScaledRotary
    for n, p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p, 'inv_freq', None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model, n, PartNTKScaledRotary(dim=dim,
                                                  original_max_position_embeddings=original_max_position_embeddings or p.max_position_embeddings,
                                                  max_position_embeddings=max_position_embeddings,
                                                  base=base,
                                                  scale=scale,
                                                  ntk_factor=ntk_factor,
                                                  extrapolation_factor=extrapolation_factor,
                                                  device=inv_freq.device))





@dataclass
class RopeBaseArguments:
    method: Optional[str] = field(default=None, metadata={"help": "one of dynamic_scaled,dynamic_part_ntk,ntk_scaled,linear_scaled,part_ntk_scaled"})
    name: Optional[str] = field(default="rotary_emb", metadata={"help": "name of rope layer"})
    base: int = field(default=10000, metadata={"help": "base default 10000"})


@dataclass
class RotaryDynamicScaledArguments(RopeBaseArguments):
    max_position_embeddings: int = field(default=2048, metadata={"help": "max_position_embeddings"})
    ntk : Optional[Union[bool,float]] = field(default=False, metadata={"help": "ntk"})


@dataclass
class RotaryDynamicPartNtkArguments(RopeBaseArguments):
    max_position_embeddings: int = field(default=2048, metadata={"help": "max_position_embeddings"})
    original_max_position_embeddings: int = field(default=2048, metadata={"help": "original_max_position_embeddings"})
    ntk_factor : float = field(default=1, metadata={"help": "ntk_factor"})
    extrapolation_factor: float = field(default=1, metadata={"help": "extrapolation_factor"})
    finetuned: bool = field(default=False, metadata={"help": "finetuned"})


@dataclass
class RotaryNtkScaledArguments(RopeBaseArguments):
    max_position_embeddings: int = field(default=2048, metadata={"help": "max_position_embeddings"})
    alpha: float = field(default=1, metadata={"help": "alpha"})



@dataclass
class RotaryLinearScaledArguments(RopeBaseArguments):
    max_position_embeddings: int = field(default=2048, metadata={"help": "max_position_embeddings"})
    scale: float = field(default=1, metadata={"help": "alpha"})


@dataclass
class RotaryPartNtkScaledArguments(RopeBaseArguments):
    original_max_position_embeddings: int = field(default=2048, metadata={"help": "original_max_position_embeddings"})
    max_position_embeddings: int = field(default=2048, metadata={"help": "max_position_embeddings"})
    scale: float = field(default=1, metadata={"help": "alpha"})
    ntk_factor: float = field(default=1, metadata={"help": "ntk_factor"})
    extrapolation_factor: float = field(default=1, metadata={"help": "extrapolation_factor"})





def inject_rope_scale_layer(model,rope_args):
    if rope_args is None:
        return None
    if isinstance(rope_args,RotaryDynamicScaledArguments):
        rope_args: RotaryDynamicScaledArguments
        patch_for_dynamic_scaled_rotary_embeddings(model,name=rope_args.name,max_position_embeddings=rope_args.max_position_embeddings,
                                               base=rope_args.base, ntk=rope_args.ntk)
    elif isinstance(rope_args,RotaryDynamicPartNtkArguments):
        rope_args: RotaryDynamicPartNtkArguments
        patch_for_dynamic_part_ntk_rotary_embeddings(model, name=rope_args.name,
                                                     max_position_embeddings=rope_args.max_position_embeddings,
                                                     original_max_position_embeddings=rope_args.original_max_position_embeddings ,
                                                     base=rope_args.base,
                                                     ntk_factor=rope_args.ntk_factor,
                                                     extrapolation_factor=rope_args.extrapolation_factor,
                                                     finetuned=rope_args.finetuned)

    elif isinstance(rope_args, RotaryNtkScaledArguments):
        rope_args: RotaryNtkScaledArguments
        patch_for_ntk_scaled_rotary_embeddings(model, name=rope_args.name,
                                               max_position_embeddings=rope_args.max_position_embeddings,
                                               base=rope_args.base,
                                               alpha=rope_args.alpha)
    elif isinstance(rope_args, RotaryLinearScaledArguments):
        rope_args: RotaryLinearScaledArguments
        patch_for_linear_scaled_rotary_embeddings(model, name=rope_args.name,
                                                  max_position_embeddings=rope_args.max_position_embeddings,
                                                  base=rope_args.base,
                                                  scale=rope_args.scale)
    elif isinstance(rope_args, RotaryPartNtkScaledArguments):
        rope_args: RotaryPartNtkScaledArguments
        patch_for_part_ntk_scaled_rotary_embeddings(model, name=rope_args.name,
                                                    original_max_position_embeddings=rope_args.original_max_position_embeddings,
                                                    max_position_embeddings=rope_args.max_position_embeddings,
                                                    base=rope_args.base,
                                                    scale=rope_args.scale,
                                                    ntk_factor=rope_args.ntk_factor,
                                                    extrapolation_factor=rope_args.extrapolation_factor)
    else:
        return None
    return model