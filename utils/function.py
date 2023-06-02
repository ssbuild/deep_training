# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/30 15:16
import copy
from dataclasses import asdict

__all__ = [
    'copy_dataclass'
]




def copy_dataclass(d_obj, **kw):
    D_class = d_obj.__class__
    input = {**kw}
    for key, value in asdict(d_obj).items():
        # If the attribute is passed to __init__
        if d_obj.__dataclass_fields__[key].init:
            input[key] = copy.deepcopy(value)

    copy_d = D_class(**input)
    return copy_d