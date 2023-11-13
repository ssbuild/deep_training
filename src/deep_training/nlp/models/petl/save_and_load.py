# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Optional
import torch
from safetensors.torch import load_file as safe_load_file
from .config.prompt import PromptType, PromptLearningConfig
from ...layers.petl.constants import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from ...layers.petl.utils import infer_device


def get_petl_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.petl_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.lora_type in ('lora', 'adalora'):
        # to_return = lora_state_dict(model, bias=model.petl_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.lora_type == "adalora":
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)
    elif config.lora_type == "ia3":
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}
    elif config.peft_type == "loha":
        to_return = {k: state_dict[k] for k in state_dict if "hada_" in k}

    elif config.peft_type == "lokr":
        to_return = {k: state_dict[k] for k in state_dict if "lokr_" in k}
    else:
        raise NotImplementedError
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_petl_model_state_dict(model, petl_model_state_dict, adapter_name="default", strict=False):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        petl_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.petl_config[adapter_name]
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in petl_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = petl_model_state_dict

    if config.lora_type in ('lora', 'adalora', 'ia3',"loha","lokr"):
        petl_model_state_dict = {}
        parameter_prefix = {
            "ia3": "ia3_",
            "lora": "lora_",
            "adalora": "lora_",
            "loha": "hada_",
            "lokr": "lokr_",
        }[config.lora_type]
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                petl_model_state_dict[k] = v
            else:
                petl_model_state_dict[k] = v
        if config.peft_type == "adalora":
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(petl_model_state_dict, strict=strict)
    return load_result






def load_petl_weights(model_id: str, device: Optional[str] = None, **kwargs) -> dict:
    r"""
    A helper method to load the effi weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, kwargs["subfolder"])
        if kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        raise ValueError(
            f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
            f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
        )

    if use_safetensors:
        adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))
    return adapters_weights









def get_prompt_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.prompt_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    if config.prompt_type == PromptType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_prompt_model_state_dict(model, prompt_model_state_dict, adapter_name="default",strict=False):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`LoraModel`]): The LoraModel.
        prompt_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.prompt_config[adapter_name]
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in prompt_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = prompt_model_state_dict


    if isinstance(config, PromptLearningConfig) or config.prompt_type == PromptType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    model.load_state_dict(peft_model_state_dict, strict=strict)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
