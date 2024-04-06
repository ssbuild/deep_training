# -*- coding: utf-8 -*-
# @Time:  23:07
# @Author: tk
# @File：export_transformers
import gc
import json
import os
from collections import OrderedDict
import torch
import safetensors.torch as st
from safetensors import safe_open
import argparse

from transformers.modeling_utils import shard_checkpoint
from transformers.utils import WEIGHTS_INDEX_NAME

parser = argparse.ArgumentParser(description='convert to huggingface.')
parser.add_argument('src',  type=str,help='src model file or path')
parser.add_argument('dst',  type=str,help='dst model path')
parser.add_argument('mode',  choices=['hf', 'safetensors'], default="safetensors", help='dst model file')
parser.add_argument('max_shard_size',default="10GB", help='max size per block')

args = parser.parse_args()


model_file: str = args.src
dest_path: str = args.dst
mode: str = args.mode
max_shard_size = args.max_shard_size

if mode == "hf":
    file_save_fn = torch.save
else:
    file_save_fn = st.save_file

def convert2hf(weight):
    w = OrderedDict()
    for k,v in weight.items():
        if k.startswith('base_model.model'):
            w['.'.join(k.split('.')[3:])] = v
        else:
            w[k] = v
    return w

def main():
    if model_file.endswith('.safetensors'):
        tensors = {}
        with safe_open(model_file, framework="pt", device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    else:
        tensors = torch.load(model_file)

    state_dict = convert2hf(tensors)

    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size)
    for shard_file, shard in shards.items():
        file_save_fn(shard, os.path.join(dest_path, shard_file))

    if index is not None:
        save_index_file = os.path.join(dest_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        # 5. Clean up shards (for some reason the file PyTorch saves take the same space as the whole state_dict
        print(
            "Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model."
        )
        shard_files = list(shards.keys())

        del state_dict
        del shards
        gc.collect()

        for shard_file in shard_files:
            state_dict = torch.load(os.path.join(dest_path, shard_file))
            file_save_fn({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(dest_path, shard_file))
    del state_dict
    gc.collect()



def __main__():
    main()






