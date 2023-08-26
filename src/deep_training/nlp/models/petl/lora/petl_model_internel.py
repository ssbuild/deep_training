# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 9:17
import logging
from abc import ABC, abstractmethod
from typing import Union, Any,Dict,AnyStr
from torch import nn

from .configuration import PetlConfig
from ...transformer_base import TransformerBase
from ....layers.petl.utils import _get_submodules, prepare_model_for_kbit_training
from ....layers.petl.petl_layer import PetlLayerAbstract

logger = logging.getLogger(__name__)

class PetlModelAbstract(nn.Module, ABC):
    r"""
    A base tuner model that provides the common methods and attributes for all tuners that are injectable into a
    torch.nn.Module

    For adding a new Tuner class, one needs to overwrite the following methods:

    - **_prepare_adapter_config**:
        A private method to eventually prepare the adapter config, for example in case the field `target_modules` is
        missing.
    - **_check_target_module_exists**:
        A helper private method to check if the passed module's key name matches any of the target modules in the
        adatper_config.
    - **_create_and_replace**:
        A private method to create and replace the target module with the adapter module.
    - **_check_target_module_exists**:
        A private helper method to check if the passed module's key name matches any of the target modules in the
        adatper_config.

    The easiest is to check what is done in the `peft.tuners.lora.LoraModel` class.

    Attributes:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        forward (`Callable`):
            The forward method of the model.
        petl_config (`Union[`PetlConfig`, dict[str, EffiConfig]]`):
            The adapter configuration object, it should be a dictionary of `str` to `EffiConfig` objects. One can also
            pass a EffiConfig object and a new adapter will be created with the default name `adapter` or create a new
            dictionary with a key `adapter_name` and a value of that peft config.
        config (`dict[str, Any]`):
            The model configuration object, it should be a dictionary of `str` to `Any` objects.
    """

    def __init__(self, model, petl_config: Union[PetlConfig, Dict[AnyStr, PetlConfig]], adapter_name: AnyStr,
                 auto_prepare_kbit_training=True,
                 use_input_require_grads=True,
                 use_gradient_checkpointing=True
                 ) -> None:
        super().__init__()

        self.auto_prepare_kbit_training = auto_prepare_kbit_training
        self.use_input_require_grads = use_input_require_grads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.model = model

        # For advanced developpers, if you want to attach multiple adapters to your
        # model, just add a `petl_config` dict attribute to your model.
        if not hasattr(self, "petl_config"):
            self.petl_config = {adapter_name: petl_config} if isinstance(petl_config, PetlConfig) else petl_config
        else:
            logger.info(
                "Already found a `petl_config` attribute in the model. This will lead to having multiple adapters"
                " in the model. Make sure to know what you are doing!"
            )
            if isinstance(petl_config, PetlConfig):
                self.petl_config[adapter_name] = petl_config
            else:
                # user is adding a dict of EffiConfigs
                self.petl_config.update(petl_config)

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

        self.inject_adapter(self.model, adapter_name)

        # Copy the petl_config in the injected model.
        self.model.petl_config = self.petl_config

    def get_transformer_model(self):
        return self.model.model if isinstance(self.model, TransformerBase) else self.model

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)


    @abstractmethod
    def _prepare_adapter_config(self, petl_config: PetlConfig, model_config: dict) -> PetlConfig:
        r"""
        A private method to eventually prepare the adapter config. For transformers based models, if
        `petl_config.target_modules` is None, we can automatically infer the target modules from the
        `TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING`. This method can be further refactored in the future to
        automatically infer it for all tuner models.

        Check out `peft.tuner.lora.LoraModel._prepare_adapter_config` for an example.

        Args:
            petl_config (`str`):
                The adapter config.
            model_config (`str`):
                The transformers model config, that config should contain the `model_type` key.
        """
        ...

    @staticmethod
    @abstractmethod
    def _check_target_module_exists(petl_config: PetlConfig, key: str) -> bool:
        r"""
        A helper private method to check if the passed module's key name matches any of the target modules in the
        `petl_config.target_modules` list. If it does, return `True`, else return `False`.

        Args:
            petl_config (`PetlConfig`):
                The adapter config.
            key (`str`):
                The module's key name.
        """
        ...

    @abstractmethod
    def _create_and_replace(
        self,
        petl_config: PetlConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optionnal_kwargs: Any,
    ) -> None:
        r"""
        Inplace replacement of the target module with the adapter layer. This method needs to be overriden by all the
        tuner classes.

        Check `peft.tuners.lora.LoraModel._create_and_replace` for an example.

        Args:
            petl_config (`PetlConfig`):
                The adapter config.
            adapter_name (`str`):
                The adapter name.
            target (`nn.Module`):
                The target module.
            target_name (`str`):
                The target module's name.
            parent (`nn.Module`):
                The parent module.
            **optionnal_kwargs (`dict`):
                The optional keyword arguments to pass to deal with particular cases (e.g. 8bit, 4bit quantization)
        """
        ...

    @abstractmethod
    def _mark_only_adapters_as_trainable(self):
        r"""
        A helper method to mark only the adapter layers as trainable (i.e. module.requires_grad = False) This needs to
        be overriden for all tuner classes to match the correct key names.

        Check `peft.tuners.lora.LoraModel._mark_only_adapters_as_trainable` for an example.
        """
        ...

    def _check_new_adapter_config(self, config: PetlConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        pass

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers. This method is called under the
        hood by `peft.mapping.get_peft_model` if a non-prompt tuning adapter class is passed.

        The corresponding PEFT config is directly retrieved from the `petl_config` attribute of the BaseTuner class.

        Args:
            model (`nn.Module`):
                The model to be tuned.
            adapter_name (`str`):
                The adapter name.
        """
        transformer_model = self.get_transformer_model()


        loaded_in_4bit = getattr(transformer_model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(transformer_model, "is_loaded_in_8bit", False)

        if self.auto_prepare_kbit_training and (loaded_in_4bit or loaded_in_8bit):
            prepare_model_for_kbit_training(transformer_model,
                                            use_input_require_grads=self.use_input_require_grads,
                                            use_gradient_checkpointing=self.use_gradient_checkpointing)


        petl_config = self.petl_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(petl_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        petl_config = self._prepare_adapter_config(petl_config, model_config)

        for key in key_list:
            if not self._check_target_module_exists(petl_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)

            optionnal_kwargs = {
                "loaded_in_8bit": loaded_in_8bit,
                "loaded_in_4bit": loaded_in_4bit,
                "current_key": key,
            }
            self._create_and_replace(petl_config, adapter_name, target, target_name, parent, **optionnal_kwargs)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {petl_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

        self._mark_only_adapters_as_trainable()

        if self.petl_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

    def merge_adapter(self):
        """
        This method merges the LoRa layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, PetlLayerAbstract):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, PetlLayerAbstract):
                module.unmerge()
