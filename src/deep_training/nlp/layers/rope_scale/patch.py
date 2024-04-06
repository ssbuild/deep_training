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
                                               base=10000, ntk=False,model_type=None):
    assert name
    if model_type is not None:
        model_type = model_type.lower()
    from .DynamicScaledRotary import DynamicScaledRotary,DynamicScaledRotaryGLM,DynamicScaledRotaryGLM2,DynamicScaledRotaryMoss
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            if model_type == 'chatglm':
                class_name = DynamicScaledRotaryGLM
            elif  model_type == 'chatglm2':
                class_name = DynamicScaledRotaryGLM2
            elif model_type == 'moss':
                class_name = DynamicScaledRotaryMoss
            else:
                class_name = DynamicScaledRotary
            rope_module = class_name(dim=dim,
                                                max_position_embeddings= max_position_embeddings or p.max_position_embeddings,
                                                ntk=ntk,
                                                base=base,
                                                device=inv_freq.device)
            setattr(model.get_submodule('.'.join(n.split('.')[:-1])), name, rope_module)

def patch_for_dynamic_part_ntk_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=2048,original_max_position_embeddings=None ,
                                                 base=10000, ntk_factor=1.0, extrapolation_factor=1.0, finetuned=False,model_type=None):
    assert name
    if model_type is not None:
        model_type = model_type.lower()
    assert model_type not in ['chatglm', 'chatglm2','moss'], ValueError('NotImplemented')
    from .DynamicPartNTKScaledRotary import DynamicPartNTKScaledRotary
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            rope_module = DynamicPartNTKScaledRotary(dim=dim,
                                       max_position_embeddings=max_position_embeddings,
                                       original_max_position_embeddings=original_max_position_embeddings or p.max_position_embeddings,
                                       base=base,
                                       ntk_factor=ntk_factor,
                                       extrapolation_factor=extrapolation_factor,
                                       finetuned=finetuned,
                                       device=inv_freq.device)
            setattr(model.get_submodule('.'.join(n.split('.')[:-1])), name, rope_module)

def patch_for_ntk_scaled_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=None, base=10000, alpha=1.0,model_type=None):
    assert name
    if model_type is not None:
        model_type = model_type.lower()
    from .NTKScaledRotary import NTKScaledRotary,NTKScaledRotaryGLM,NTKScaledRotaryGLM2,NTKScaledRotaryMoss
    for n,p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            if model_type == 'chatglm':
                class_name = NTKScaledRotaryGLM
            elif  model_type == 'chatglm2':
                class_name = NTKScaledRotaryGLM2
            elif model_type == 'moss':
                class_name = NTKScaledRotaryMoss
            else:
                class_name = NTKScaledRotary
            rope_module = class_name(dim=dim,
                                    max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                                    base=base,
                                    alpha=alpha,
                                    device=inv_freq.device)

            setattr(model.get_submodule('.'.join(n.split('.')[:-1])), name, rope_module)


def patch_for_linear_scaled_rotary_embeddings(model,name='rotary_emb', max_position_embeddings=None, base=10000, scale=1.0,model_type=None):
    assert name
    if model_type is not None:
        model_type = model_type.lower()
    from .LinearScaledRotary import LinearScaledRotary,LinearScaledRotaryGLM,LinearScaledRotaryGLM2,LinearScaledRotaryMoss
    for n, p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p, 'inv_freq', None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            if model_type == 'chatglm':
                class_name = LinearScaledRotaryGLM
            elif  model_type == 'chatglm2':
                class_name = LinearScaledRotaryGLM2
            elif model_type == 'moss':
                class_name = LinearScaledRotaryMoss
            else:
                class_name = LinearScaledRotary
            rope_module = class_name(dim=dim,
                               max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                               base=base,
                               scale=scale,
                               device=inv_freq.device)
            setattr(model.get_submodule('.'.join(n.split('.')[:-1])), name, rope_module)

def patch_for_part_ntk_scaled_rotary_embeddings(model,name='rotary_emb', original_max_position_embeddings=None,max_position_embeddings=2048,
                                                base=10000, scale=1.0, ntk_factor=1.0, extrapolation_factor=1.0,model_type=None):
    assert name
    if model_type is not None:
        model_type = model_type.lower()
    assert model_type not in ['chatglm', 'chatglm2','moss'], ValueError('NotImplemented')
    from .PartNTKScaledRotary import PartNTKScaledRotary
    for n, p in model.named_modules():
        if n.endswith(name):
            inv_freq: nn.Module = getattr(p, 'inv_freq', None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            rope_module = PartNTKScaledRotary(dim=dim,
                                original_max_position_embeddings=original_max_position_embeddings or p.max_position_embeddings,
                                max_position_embeddings=max_position_embeddings,
                                base=base,
                                scale=scale,
                                ntk_factor=ntk_factor,
                                extrapolation_factor=extrapolation_factor,
                                device=inv_freq.device)
            setattr(model.get_submodule('.'.join(n.split('.')[:-1])), name, rope_module)



@dataclass
class RopeBaseArguments:
    model_type: Optional[str] = field(default=None, metadata={"help": "name of model"})
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
                                                   base=rope_args.base, ntk=rope_args.ntk,
                                                   model_type=rope_args.model_type)
    elif isinstance(rope_args,RotaryDynamicPartNtkArguments):
        rope_args: RotaryDynamicPartNtkArguments
        patch_for_dynamic_part_ntk_rotary_embeddings(model, name=rope_args.name,
                                                     max_position_embeddings=rope_args.max_position_embeddings,
                                                     original_max_position_embeddings=rope_args.original_max_position_embeddings ,
                                                     base=rope_args.base,
                                                     ntk_factor=rope_args.ntk_factor,
                                                     extrapolation_factor=rope_args.extrapolation_factor,
                                                     finetuned=rope_args.finetuned,
                                                     model_type=rope_args.model_type)

    elif isinstance(rope_args, RotaryNtkScaledArguments):
        rope_args: RotaryNtkScaledArguments
        patch_for_ntk_scaled_rotary_embeddings(model, name=rope_args.name,
                                               max_position_embeddings=rope_args.max_position_embeddings,
                                               base=rope_args.base,
                                               alpha=rope_args.alpha,
                                               model_type=rope_args.model_type)
    elif isinstance(rope_args, RotaryLinearScaledArguments):
        rope_args: RotaryLinearScaledArguments
        patch_for_linear_scaled_rotary_embeddings(model, name=rope_args.name,
                                                  max_position_embeddings=rope_args.max_position_embeddings,
                                                  base=rope_args.base,
                                                  scale=rope_args.scale,
                                                  model_type=rope_args.model_type)
    elif isinstance(rope_args, RotaryPartNtkScaledArguments):
        rope_args: RotaryPartNtkScaledArguments
        patch_for_part_ntk_scaled_rotary_embeddings(model, name=rope_args.name,
                                                    original_max_position_embeddings=rope_args.original_max_position_embeddings,
                                                    max_position_embeddings=rope_args.max_position_embeddings,
                                                    base=rope_args.base,
                                                    scale=rope_args.scale,
                                                    ntk_factor=rope_args.ntk_factor,
                                                    extrapolation_factor=rope_args.extrapolation_factor,
                                                    model_type=rope_args.model_type)
    else:
        return None
    return model