# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/7 14:01
from functools import partial
import transformers
from transformers import TOKENIZER_MAPPING,CONFIG_MAPPING,AutoTokenizer,AutoConfig


def _config_register(self, key, value, exist_ok=False):
    """
    Register a new configuration in this mapping.
    """
    if key in self._mapping.keys() and not exist_ok:
        raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
    self._extra_content[key] = value


def _model_register(self, key, value, exist_ok=False):
    """
    Register a new model in this mapping.
    """
    if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers model.")
    self._extra_content[key] = value


def _parse_transformer_version():
    ver = transformers.__version__.split('.')
    ver = [int(_) for _ in ver]
    return ver


def register_transformer_config(config_class, exist_ok=True):
    ver = _parse_transformer_version()
    old_fn_back = None
    if ver[0] <= 4 and ver[1] <= 30:
        old_fn_back = CONFIG_MAPPING.register
        CONFIG_MAPPING.register = _config_register
        register = partial(CONFIG_MAPPING.register, CONFIG_MAPPING)
    else:
        register = CONFIG_MAPPING.register
    register(config_class.model_type, config_class, exist_ok=exist_ok)
    if old_fn_back:
        CONFIG_MAPPING.register = old_fn_back

def register_transformer_model(model_class,transormer_auto_class, exist_ok=True):
    ver = _parse_transformer_version()
    old_fn_back = None
    if ver[0] <= 4 and ver[1] <= 30:
        old_fn_back = transormer_auto_class._model_mapping.register
        transormer_auto_class._model_mapping.register = _model_register
        self = transormer_auto_class._model_mapping
        register = partial(transormer_auto_class._model_mapping.register, self)
    else:
        register = transormer_auto_class._model_mapping.register
    register(model_class.config_class,model_class,exist_ok=exist_ok)
    if old_fn_back:
        transormer_auto_class._model_mapping.register = old_fn_back


def register_transformer_tokenizer(tokenizer_class,slow_tokenizer_class, fast_tokenizer_class, exist_ok=True):
    ver = _parse_transformer_version()
    old_fn_back = None
    if ver[0] <= 4 and ver[1] <= 30:
        old_fn_back = TOKENIZER_MAPPING.register
        TOKENIZER_MAPPING.register = _config_register
        self = TOKENIZER_MAPPING
        register = partial(TOKENIZER_MAPPING.register, self)
    else:
        register = TOKENIZER_MAPPING.register
    register(tokenizer_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)
    if old_fn_back:
        TOKENIZER_MAPPING.register = old_fn_back
