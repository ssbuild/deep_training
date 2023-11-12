# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 9:13
import re
import warnings
from typing import Union, List, Tuple
from .utils import COMMON_LAYERS_PATTERN


class PetlLayerBase:
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """
    active_adapter = None

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names: Tuple[str ] = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: Tuple[str ] = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter: Union[ str, List[ str ] ] = "default"

    # List all merged adapters
    merged_adapters: List[str ] = []

    def merge(self, *args) -> None:
        raise NotImplementedError

    def unmerge(self, *args) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names: Union[ str, List[ str ] ]):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> List[str ]:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])




def check_target_module_exists(config, key: str) -> Union[ bool, re.Match[ str ], None ]:
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules or any(
            key.endswith(f".{target_key}") for target_key in config.target_modules
        )
        is_using_layer_indexes = getattr(config, "layers_to_transform", None) is not None
        layer_indexing_pattern = getattr(config, "layers_pattern", None)

        if is_using_layer_indexes and target_module_found:
            layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
            layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

            for pattern in layers_pattern:
                layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                if layer_index is not None:
                    layer_index = int(layer_index.group(1))
                    if isinstance(config.layers_to_transform, int):
                        target_module_found = layer_index == config.layers_to_transform
                    else:
                        target_module_found = layer_index in config.layers_to_transform

                    break
                else:
                    target_module_found = False
    return target_module_found

