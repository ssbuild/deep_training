# -*- coding: utf-8 -*-
# @Time:  22:06
# @Author: tk
# @File：wrapper
import yaml
import os

def load_yaml(filename):
    base_dir = os.path.abspath(os.path.dirname(filename))
    with open(filename, mode='r', encoding='utf-8') as f:
        cfg = yaml.full_load(f)
    for inc in cfg.pop("includes", []):
        if not os.path.exists(inc):
            inc = os.path.join(base_dir, inc)
        with open(inc, mode='r', encoding='utf-8') as f:
            cfg.update(yaml.full_load(f))
    return cfg
