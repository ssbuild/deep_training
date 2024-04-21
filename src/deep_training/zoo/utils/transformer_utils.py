# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 9:18
import typing
def hf_decorator(fn):
    def hf_fn(self,*args,**kwargs):
        preprocess_hf_kwargs(self,kwargs)
        fn(self,*args,**kwargs)
    return hf_fn

def preprocess_hf_kwargs(self,kwargs: typing.Dict):
    # load_in_8bit = kwargs.get('load_in_8bit', False)
    # load_in_4bit = kwargs.get('load_in_4bit', False)
    # quantization_config = kwargs.get("quantization_config", None)
    # if quantization_config:
    #     load_in_4bit = load_in_4bit or quantization_config.load_in_4bit
    #     load_in_8bit = load_in_8bit or quantization_config.load_in_8bit
    # if not load_in_8bit and not load_in_4bit:
    #     kwargs.pop("device_map", None)
    #     kwargs.pop("quantization_config", None)
    kwargs.pop("use_input_require_grads",None)
    self.pad_to_multiple_of = kwargs.pop('pad_to_multiple_of', None)
    self.auto_prepare_kbit_training = kwargs.pop('auto_prepare_kbit_training', True)
    self.use_gradient_checkpointing = kwargs.pop('use_gradient_checkpointing', False)

