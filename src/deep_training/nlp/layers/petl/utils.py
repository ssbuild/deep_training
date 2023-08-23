# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 8:36
import copy
import importlib
import warnings
from typing import Optional

import torch
from accelerate.utils import is_xpu_available, is_npu_available


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available():
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")


def is_auto_gptq_available():
    return importlib.util.find_spec("auto_gptq") is not None


def is_optimum_available():
    return importlib.util.find_spec("optimum") is not None




# needed for prefix-tuning of bloom model
def bloom_model_postprocess_past_key_value(past_key_values):
    past_key_values = torch.cat(past_key_values)
    total_layers, batch_size, num_attention_heads, num_virtual_tokens, head_dim = past_key_values.shape
    keys = past_key_values[: total_layers // 2]
    keys = keys.transpose(2, 3).reshape(
        total_layers // 2, batch_size * num_attention_heads, head_dim, num_virtual_tokens
    )
    values = past_key_values[total_layers // 2 :]
    values = values.reshape(total_layers // 2, batch_size * num_attention_heads, num_virtual_tokens, head_dim)

    return tuple(zip(keys, values))


def prepare_model_for_kbit_training(model,
                                    use_input_require_grads=True,
                                    use_gradient_checkpointing=True):
    r"""
       This method wraps the entire protocol for preparing a model before running a training. This includes:
           1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
           head to fp32

       Args:
           model, (`transformers.PreTrainedModel`):
               The loaded model from `transformers`
       """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_input_require_grads:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if loaded_in_kbit and use_gradient_checkpointing:

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def prepare_model_for_int8_training(*args, **kwargs):
    warnings.warn(
        "prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.",
        FutureWarning,
    )
    return prepare_model_for_kbit_training(*args, **kwargs)

# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids




class ModulesToSaveWrapper(torch.nn.Module):
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self.update(adapter_name)
        self.active_adapter = adapter_name

    def update(self, adapter_name):
        self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))


    def forward(self, *args, **kwargs):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module(*args, **kwargs)

        if not torch.is_autocast_enabled():
            dtype = self.modules_to_save[self.active_adapter].weight.dtype
            args = (_.to(dtype) for _ in args)
            for k in kwargs:
                kwargs[k] = kwargs[k].to(dtype)

        return self.modules_to_save[self.active_adapter](*args, **kwargs)


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
            else:
                for param in target.parameters():
                    param.requires_grad = True
                setattr(parent, target_name, ModulesToSaveWrapper(target, adapter_name))

    for k, n in model.named_modules():
        if isinstance(n,ModulesToSaveWrapper):
            for p in n.original_module.parameters():
                p.requires_grad = False


def _set_adapter(model, adapter_name):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            module.active_adapter = adapter_name


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def _is_valid_match(key: str, target_key: str):
    """
    Helper function to match module names target_key and key. Makes sure that either the key is exactly the target_key
    or the target_key is a submodule of key
    """
    if key.endswith(target_key):
        if len(key) > len(target_key):
            return key.endswith("." + target_key)  # must be a sub module
        return True
    return False


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size

def get_quantization_config(model: torch.nn.Module, method: str):
    """
    Get the quantization config of the related quantization method
    """
    if (
        hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
        and (getattr(model, "quantization_method", None) == method)
    ):
        return model.config.quantization_config
    return None

def get_auto_gptq_quant_linear(gptq_quantization_config):
    """
    Get the right AutoGPTQQuantLinear class based on the quantization config file
    """
    if is_auto_gptq_available():
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        if gptq_quantization_config is not None:
            desc_act = gptq_quantization_config.desc_act
            group_size = gptq_quantization_config.group_size
            bits = gptq_quantization_config.bits
            disable_exllama = gptq_quantization_config.disable_exllama
            AutoGPTQQuantLinear = dynamically_import_QuantLinear(
                use_triton=False,
                desc_act=desc_act,
                group_size=group_size,
                bits=bits,
                disable_exllama=disable_exllama,
            )
            return AutoGPTQQuantLinear
    return None


# Get current device name based on available devices
def infer_device():
    if torch.cuda.is_available():
        torch_device = "cuda"
    elif is_xpu_available():
        torch_device = "xpu"
    elif is_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
}

TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gpt2": ["c_attn"],
    # "bloom": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gptj": ["q_proj", "v_proj"],
    # "gpt_neox": ["query_key_value"],
    # "gpt_neo": ["q_proj", "v_proj"],
    # "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    # "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
}

TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
}

# WEIGHTS_NAME = "adapter_model.bin"
# CONFIG_NAME = "adapter_config.json"
