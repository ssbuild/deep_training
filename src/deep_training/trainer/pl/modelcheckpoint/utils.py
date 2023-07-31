# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/1 9:10
import copy
import os.path
import typing
import torch

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

except:
    deepspeed = None
    ZeroParamStatus = None

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def gather_ds_state_dict(checkpoints: typing.Dict,output_filename,zero_stage_3,is_global_zero,config):
    dirname = os.path.dirname(output_filename)
    basename = os.path.basename(output_filename)
    for adapter_name, state in checkpoints.items():
        lora_or_prompt_config = state['config']
        state_dict = state['state_dict']
        if zero_stage_3:
            output_state_dict = {}
            for k,v in state_dict.items():
                if hasattr(v, 'ds_id'):
                    with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                           enabled=zero_stage_3):
                        v_p = v.data.cpu()
                else:
                    v_p = v.cpu()
                output_state_dict[k] = v_p
        else:
            output_state_dict = copy.copy(state_dict)

        if is_global_zero:
            if adapter_name != 'default':
                basename = adapter_name + '-' + basename
            pathfile = os.path.join(dirname,basename)
            if not os.path.exists(pathfile):
                os.mkdir(pathfile)
            filepath_new = os.path.join(dirname, basename)
            torch.save(output_state_dict, os.path.join(filepath_new,'adapter_model.bin'))
            lora_or_prompt_config.save_pretrained(filepath_new)

            config_path = os.path.join(filepath_new, 'config.json')
            if not os.path.exists(config_path):
                config.save_pretrained(filepath_new)
        del output_state_dict

