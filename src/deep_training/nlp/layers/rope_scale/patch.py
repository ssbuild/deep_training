from torch import nn

__all__ = [
    "patch_for_dynamic_scaled_rotary_embeddings",
    "patch_for_dynamic_part_ntk_rotary_embeddings",
    "patch_for_ntk_scaled_rotary_embeddings",
    "patch_for_linear_scaled_rotary_embeddings",
    "patch_for_part_ntk_scaled_rotary_embeddings",
]

def patch_for_dynamic_scaled_rotary_embeddings(model,max_position_embeddings=None,
                                               base=10000, ntk=False):
    from .DynamicScaledRotary import DynamicScaledRotary
    for n,p in model.named_modules():
        if n.endswith('rotary_emb'):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model,n,DynamicScaledRotary(dim=dim,
                                                max_position_embeddings= max_position_embeddings or p.max_position_embeddings,
                                                ntk=ntk,
                                                base=base,
                                                device=inv_freq.device))

def patch_for_dynamic_part_ntk_rotary_embeddings(model, max_position_embeddings=2048,original_max_position_embeddings=None ,
                                                 base=10000, ntk_factor=1, extrapolation_factor=1, finetuned=False):
    from .DynamicPartNTKScaledRotary import DynamicPartNTKScaledRotary
    for n,p in model.named_modules():
        if n.endswith('rotary_emb'):
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

def patch_for_ntk_scaled_rotary_embeddings(model, max_position_embeddings=None, base=10000, alpha=1):
    from .NTKScaledRotary import NTKScaledRotary
    for n,p in model.named_modules():
        if n.endswith('rotary_emb'):
            inv_freq: nn.Module = getattr(p,'inv_freq',None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model,n,NTKScaledRotary(dim=dim,
                                            max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                                            base=base,
                                            alpha=alpha,
                                            device=inv_freq.device))

def patch_for_linear_scaled_rotary_embeddings(model, max_position_embeddings=None, base=10000, scale=1):
    from .LinearScaledRotary import LinearScaledRotary
    for n, p in model.named_modules():
        if n.endswith('rotary_emb'):
            inv_freq: nn.Module = getattr(p, 'inv_freq', None)
            if inv_freq is None:
                continue
            dim = inv_freq.size(-1) * 2
            setattr(model, n, LinearScaledRotary(dim=dim,
                                              max_position_embeddings=max_position_embeddings or p.max_position_embeddings,
                                              base=base,
                                              scale=scale,
                                              device=inv_freq.device))

def patch_for_part_ntk_scaled_rotary_embeddings(model, original_max_position_embeddings=None,max_position_embeddings=2048,
                                                base=10000, scale=1, ntk_factor=1, extrapolation_factor=1):
    from .PartNTKScaledRotary import PartNTKScaledRotary
    for n, p in model.named_modules():
        if n.endswith('rotary_emb'):
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

