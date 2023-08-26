# -*- coding: utf-8 -*-
# @Time:  10:42
# @Author: tk
# @File：lora_wrapper
import copy
import os
from typing import Optional,Tuple,Dict,List,Any,Union,Callable
from contextlib import contextmanager
import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin
from safetensors.torch import save_file as safe_save_file
from .....utils.function import copy_dataclass
from ....layers.petl.utils import _set_trainable, _set_adapter, infer_device
from .configuration import WEIGHTS_NAME, LoraConfig, AdaLoraConfig, PetlConfig, IA3Config, PetlArguments, \
    SAFETENSORS_WEIGHTS_NAME
from .lora_model import LoraModule
from .adalora_model import AdaLoraModule
from .ia3_model import IA3Module
from .save_and_load import get_lora_model_state_dict, set_lora_model_state_dict, load_petl_weights

LORA_TYPE_TO_MODEL_MAPPING = {
    "ia3": IA3Module,
    "lora": LoraModule,
    "adalora": AdaLoraModule,
}

LORA_TYPE_TO_CONFIG_MAPPING = {
    "ia3": IA3Config,
    "lora": LoraConfig,
    "adalora": AdaLoraConfig,
}

class PetlModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Lora methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Lora.
        petl_config ([`LoraConfig`]): The configuration of the Lora model.


    **Attributes**:
        - **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer model used for Lora.
        - **lora_config** ([`LoraConfig`]) -- The configuration of the Lora model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Lora if
        using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Lora if
        using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model, petl_config: PetlConfig, adapter_name="default",
                 auto_prepare_kbit_training=True,
                 use_input_require_grads=True,
                 use_gradient_checkpointing=True):
        '''
            model TransformerBase , model.model
        '''
        super().__init__()
        assert petl_config.lora_type is not None
        self.base_model = model
        self.config = getattr(model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.petl_config = {}
        self.active_adapter = adapter_name
        self.lora_type = petl_config.lora_type
        self.base_model_torch_dtype = getattr(model, "dtype", None)
        self.petl_config[adapter_name] = petl_config
        self.base_model: LoraModule = LORA_TYPE_TO_MODEL_MAPPING[petl_config.lora_type](
            self.base_model, self.petl_config, adapter_name,
            auto_prepare_kbit_training=auto_prepare_kbit_training,
            use_input_require_grads=use_input_require_grads,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.set_additional_trainable_modules(petl_config, adapter_name)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1


    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.petl_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.petl_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.petl_config.keys())} - got {selected_adapters}."
                )

        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            petl_config = self.petl_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_lora_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if safe_serialization:
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if petl_config.base_model_name_or_path is None:
                petl_config.base_model_name_or_path = (
                    self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = petl_config.inference_mode
            petl_config.inference_mode = True


            auto_mapping_dict = None

            petl_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            petl_config.inference_mode = inference_mode

   

    def get_all_state_dict(self, **kwargs):
        checkpoints = {}
        for adapter_name, lora_config in self.petl_config.items():
            lora_config: LoraConfig
            # save only the trainable weights
            output_state_dict = get_lora_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )

            if lora_config.base_model_name_or_path is None:
                lora_config.base_model_name_or_path = (
                    self.base_model.model.__dict__.get("model_name_or_path", None)
                )
            inference_mode = lora_config.inference_mode
            lora_config.inference_mode = True
            checkpoints[adapter_name] = {
                "state_dict": output_state_dict,
                "config": copy_dataclass(lora_config),
            }
            lora_config.inference_mode = inference_mode
        return checkpoints
    
    @classmethod
    def from_pretrained(cls, model,
                        pretrained_model_name_or_path,
                        lora_config: PetlConfig = None,
                        adapter_name: str= "default",
                        is_trainable: bool =False, **kwargs):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """

        # load the config
        if lora_config is None:
            lora_config = LORA_TYPE_TO_CONFIG_MAPPING[
                LoraConfig.from_pretrained(pretrained_model_name_or_path, subfolder=kwargs.get("subfolder", None)).lora_type
            ].from_pretrained(pretrained_model_name_or_path, subfolder=kwargs.get("subfolder", None))
        elif isinstance(lora_config, PetlConfig):
            lora_config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PetlConfig, got {lora_config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
                set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        lora_config.inference_mode = not is_trainable
        model = cls(model, lora_config, adapter_name)
        model.load_adapter(pretrained_model_name_or_path, adapter_name, **kwargs)
        return model

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)  # defer to nn.Module's logic



    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """

        self.base_model.disable_adapter_layers()
        yield
        self.base_model.enable_adapter_layers()

    def get_base_model(self)-> Union[nn.Module,Any]:
        """
        Returns the base model.
        """
        return self.base_model.model

    def add_adapter(self, adapter_name, lora_config):
        if lora_config.lora_type != self.lora_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.lora_type} and {lora_config.lora_type}."
            )
        self.petl_config[adapter_name] = lora_config
        self.base_model.inject_adapter(self, adapter_name)
        self.set_additional_trainable_modules(lora_config, adapter_name)


    def set_additional_trainable_modules(self, lora_config, adapter_name):
        if getattr(lora_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(lora_config.modules_to_save)
            else:
                self.modules_to_save = self.modules_to_save.update(lora_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def load_adapter(self, model_id, adapter_name,
                     config=None,
                     is_trainable=False, strict=False,
                     map_preprocess: Optional[Callable]=None,**kwargs):

        torch_device = infer_device()
        if adapter_name not in self.petl_config:
            if config is None:
                # load the config
                lora_config = LORA_TYPE_TO_CONFIG_MAPPING[
                    LoraConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).lora_type
                ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            else:
                lora_config = config
            lora_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, lora_config)


        adapters_weights = load_petl_weights(model_id,device=torch_device)

        if 'state_dict' in adapters_weights:
            adapters_weights = adapters_weights['state_dict']
        if map_preprocess is not None:
            adapters_weights = map_preprocess(adapters_weights)
        # load the weights into the model
        load_result = set_lora_model_state_dict(self, adapters_weights, adapter_name=adapter_name,strict=strict)
        return load_result


    def set_adapter(self, adapter_name):
        """
        Sets the active adapter.
        """
        if adapter_name not in self.petl_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def active_effi_config(self):
        return self.petl_config[self.active_adapter]