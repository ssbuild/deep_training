# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 8:36
import torch
import copy
import importlib
import inspect
import warnings
from typing import Optional, Tuple
import importlib_metadata
import packaging
import accelerate
from accelerate.hooks import remove_hook_from_module, add_hook_to_module
from accelerate.utils import is_xpu_available, is_npu_available
from safetensors.torch import storage_ptr, storage_size
from transformers import is_torch_tpu_available

from .constants import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                        TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                        TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
                        TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                        COMMON_LAYERS_PATTERN)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_bnb_4bit_available():
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")


def is_auto_gptq_available():
    if importlib.util.find_spec("auto_gptq") is not None:
        AUTOGPTQ_MINIMUM_VERSION = packaging.version.parse("0.5.0")
        version_autogptq = packaging.version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION <= version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, "
                f"but only versions above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


def is_optimum_available():
    return importlib.util.find_spec("optimum") is not None




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


# needed for prefix-tuning of StarCoder models
def starcoder_model_postprocess_past_key_value(past_key_values):
    result = []
    for k in past_key_values:
        k = k[:, :, 0]
        k = k.permute([1, 2, 0, 3])
        k = k.reshape(*k.shape[:-2], -1)
        result.append(k)
    return tuple(result)



def prepare_model_for_kbit_training(model,gradient_checkpointing=True,gradient_checkpointing_kwargs=None,**kwargs):
    r"""
       This method wraps the entire protocol for preparing a model before running a training. This includes:
           1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
           head to fp32

       Args:
           model, (`transformers.PreTrainedModel`):
               The loaded model from `transformers`
       """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    if not loaded_in_kbit:
        s_model = getattr(model,"model",None)
        if s_model:
            loaded_in_kbit = getattr(s_model, "is_loaded_in_8bit", False) or getattr(s_model, "is_loaded_in_4bit", False)

    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
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
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def update(self, adapter_name):
        self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))

        if hasattr(self.modules_to_save[ adapter_name ], "_hf_hook"):
            old_hook = self.modules_to_save[ adapter_name ]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[ adapter_name ])
            add_hook_to_module(self.modules_to_save[ adapter_name ], new_hook)

        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[ adapter_name ].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        r"""
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[ k ] = old_hook_attr[ k ]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def forward(self, *args, **kwargs):
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save):
            return self.original_module(*args, **kwargs)

        if not torch.is_autocast_enabled():
            dtype = self.modules_to_save[self.active_adapter].weight.dtype
            args = (_.to(dtype) for _ in args)
            for k in kwargs:
                kwargs[k] = kwargs[k].to(dtype)

        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f"Adapter {adapter_name} not found in {self.modules_to_save.keys()}")

        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


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
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)


def _set_adapter(model, adapter_name):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            module.set_adapter(adapter_name)


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



def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.

    This method is the exact same copy of
    https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L282C1-L300C58 but we added
    it here manually to avoid import issue with old versions of transformers.
    """
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)


TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
    "gpt_bigcode": starcoder_model_postprocess_past_key_value,
}



def _prepare_prompt_learning_config(prompt_config, model_config):
    if prompt_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "num_layer" in model_config:
            num_layers = model_config["num_layer"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `prompt_config`")
        prompt_config.num_layers = num_layers

    if prompt_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `prompt_config`")
        prompt_config.token_dim = token_dim

    if prompt_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `prompt_config`")
        prompt_config.num_attention_heads = num_attention_heads

    if getattr(prompt_config, "encoder_hidden_size", None) is None:
        setattr(prompt_config, "encoder_hidden_size", prompt_config.token_dim)

    return prompt_config
