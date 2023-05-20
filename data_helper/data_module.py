# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 13:31

from transformers import AutoTokenizer, AutoConfig, CONFIG_MAPPING, PretrainedConfig

__all__ = [
    'load_tokenizer',
    'load_configure'
]

def load_tokenizer(tokenizer_name,
                   model_name_or_path=None,
                   class_name = None,
                   cache_dir="",
                   do_lower_case=None,
                   use_fast_tokenizer=True,
                   model_revision="main",
                   use_auth_token=None,
                   **kwargs):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        **kwargs
    }
    if do_lower_case is not None:
        tokenizer_kwargs['do_lower_case'] = do_lower_case

    if use_fast_tokenizer is not None:
        tokenizer_kwargs['use_fast'] = use_fast_tokenizer

    if class_name is not None:
        tokenizer = class_name.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    elif tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer

def load_configure(config_name,
                   model_name_or_path=None,
                   class_name = None,
                   cache_dir="",
                   model_revision="main",
                   use_auth_token=None,
                   model_type=None,
                   config_overrides=None,
                   bos_token_id=None,
                   pad_token_id=None,
                   eos_token_id=None,
                   sep_token_id=None,
                   return_dict=False,
                   task_specific_params=None,
                   **kwargs):
    config_kwargs = {
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": True if use_auth_token else None,
        "return_dict": return_dict,
        **kwargs
    }
    tmp_kwargs = {
        "bos_token_id": bos_token_id,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "sep_token_id": sep_token_id,
        "task_specific_params": task_specific_params,
    }
    for k in list(tmp_kwargs.keys()):
        if tmp_kwargs[k] is None:
            tmp_kwargs.pop(k)
    if tmp_kwargs:
        config_kwargs.update(tmp_kwargs)

    if class_name is not None:
        config = class_name.from_pretrained(config_name, **config_kwargs)
    elif isinstance(config_name,PretrainedConfig):
        for k,v in config_kwargs.items():
            setattr(config_name,k,v)
        config = config_name

    elif config_name:
        config = AutoConfig.from_pretrained(config_name, **config_kwargs)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    elif model_type:
        config = CONFIG_MAPPING[model_type].from_pretrained(model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config_gpt2 from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --config_name."
        )
    if config_overrides is not None:
        config.update_from_string(config_overrides)
    return config
