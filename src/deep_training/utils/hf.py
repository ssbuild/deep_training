# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/7 14:01
from functools import partial
import transformers
from transformers import TOKENIZER_MAPPING, CONFIG_MAPPING, AutoTokenizer, AutoConfig, PROCESSOR_MAPPING, \
    requires_backends
from transformers.image_processing_utils import BatchFeature
from transformers.utils import is_torch_dtype, is_torch_device


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


def register_transformer_processer(config_class,processor_class, exist_ok=True):
    ver = _parse_transformer_version()
    old_fn_back = None
    if ver[0] <= 4 and ver[1] <= 30:
        old_fn_back = PROCESSOR_MAPPING.register
        PROCESSOR_MAPPING.register = _config_register
        self = PROCESSOR_MAPPING
        register = partial(PROCESSOR_MAPPING.register, self)
    else:
        register = PROCESSOR_MAPPING.register
    register(config_class.model_type, processor_class, exist_ok=exist_ok)
    if old_fn_back:
        PROCESSOR_MAPPING.register = old_fn_back


class BatchFeatureDetr(BatchFeature):
    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch  # noqa

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            if k == 'labels':
                new_data[k] = []
                labels = new_data[k]
                for label in v:
                    label_new = {}
                    for sub_k,sub_v in label.items():
                        if torch.is_floating_point(sub_v):
                            # cast and send to device
                            label_new[sub_k] = sub_v.to(*args, **kwargs)
                        elif device is not None:
                            label_new[sub_k] = sub_v.to(device=device)
                        else:
                            label_new[sub_k] = sub_v
                    labels.append(label_new)
            else:
                # check if v is a floating point
                if torch.is_floating_point(v):
                    # cast and send to device
                    new_data[k] = v.to(*args, **kwargs)
                elif device is not None:
                    new_data[k] = v.to(device=device)
                else:
                    new_data[k] = v
        self.data = new_data
        return self