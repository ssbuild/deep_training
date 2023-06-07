# -*- coding: utf-8 -*-
# @Time:  22:06
# @Author: tk
# @File：pytorch

import os
import re
from collections import OrderedDict

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from .utils import gather_ds_state_dict

__all__ = [
    'ModelCheckpointEx',
]

class ModelCheckpointEx(ModelCheckpoint):
    def __init__(self,*args,**kwargs):
        self.lora_args = kwargs.pop('lora_args',None)
        self.prompt_args = kwargs.pop('prompt_args', None)
        super().__init__(*args,**kwargs)

        if self.lora_args or self.prompt_args:
            self.CHECKPOINT_NAME_LAST = "last"
            self.FILE_EXTENSION = ""

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if isinstance(trainer.strategy,DeepSpeedStrategy):
            if self.lora_args or self.prompt_args:
                self.CHECKPOINT_NAME_LAST = "last"
            self.FILE_EXTENSION = ""

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        bHandled = False
        if self.lora_args or self.prompt_args:
            bHandled = True
            model = trainer.strategy.lightning_module
            if isinstance(trainer.strategy, DeepSpeedStrategy):
                checkpoints = model.backbone.get_all_state_dict()
                gather_ds_state_dict(checkpoints,
                                     filepath,
                                     zero_stage_3=trainer.strategy.zero_stage_3,
                                     is_global_zero=trainer.strategy.is_global_zero,
                                     config=model.backbone.config)
            else:

                if trainer.is_global_zero:
                    model.backbone.save_pretrained(filepath)
                    config_path = os.path.join(filepath, 'config.json')
                    if not os.path.exists(config_path):
                        model.backbone.config.save_pretrained(filepath)

                # trainer.strategy.barrier()
        if not bHandled:
            super()._save_checkpoint(trainer,filepath)









    # def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
    #     bHandled = False
    #     if self.lora_args or self.prompt_args:
    #         bHandled = True
    #         model = trainer.strategy.lightning_module
    #         checkpoints = model.backbone.get_all_state_dict()
    #         m = trainer.strategy.model.module if isinstance(trainer.strategy, DeepSpeedStrategy) else model
    #         eng = trainer.strategy.model
    #         for adapter_name, state in checkpoints.items():
    #             lora_or_prompt_config = state['config']
    #             def state_dict_fn(module, destination, prefix, local_metadata):
    #                 return state['state_dict']
    #             h = m._register_state_dict_hook(state_dict_fn)
    #             basename = os.path.basename(filepath)
    #             dirname = os.path.dirname(filepath)
    #             if adapter_name != 'default':
    #                 basename = adapter_name + '_' + basename
    #                 config_dir = os.path.join(dirname, adapter_name)
    #             else:
    #                 config_dir = dirname
    #             filepath_new = os.path.join(dirname, basename)
    #             fn_old = None
    #             get_zero_param_shapes_old = None
    #             if isinstance(trainer.strategy, DeepSpeedStrategy):
    #                 fn_old = eng.zero_optimization_partition_gradients
    #                 eng.zero_optimization_partition_gradients = lambda: False
    #                 zero_optimizer_state = eng.zero_optimization() or eng.bfloat16_enabled()
    #                 if zero_optimizer_state:
    #                     results_list_new = []
    #                     results_list = eng._get_zero_param_shapes()
    #                     for results in results_list:
    #                         sub_module = OrderedDict()
    #                         for key,value in results.items():
    #                             key = re.sub(r'_forward_module\._TransformerLightningModule__backbone\.', '', key)
    #                             if self.lora_args:
    #                                 k1,k2 = key.split('.lora_')
    #                                 k2 = re.sub(re.compile('\.{}\.'.format(adapter_name)),'.',k2)
    #                                 key = k1 + '.lora_' + k2
    #                                 key = key.replace("modules_to_save.", "")
    #                             else:
    #                                 key = key.replace("modules_to_save.", "")
    #                                 key = key.replace(f".{adapter_name}", "")
    #                                 key = key.replace('prompt_encoder.embedding.weight','prompt_embeddings')
    #                             sub_module[key] = value
    #                         results_list_new.append(sub_module)
    #                     get_zero_param_shapes_old = eng._get_zero_param_shapes
    #                     eng._get_zero_param_shapes = lambda : results_list_new
    #             super()._save_checkpoint(trainer, filepath_new)
    #             if fn_old:
    #                 eng.zero_optimization_partition_gradients = fn_old
    #                 eng._get_zero_param_shapes = get_zero_param_shapes_old
    #             h.remove()
    #             lora_or_prompt_config.save_pretrained(config_dir)
    #
    #     if not bHandled:
    #         super()._save_checkpoint(trainer,filepath)



