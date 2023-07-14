# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/12 12:18
import gc
import json
import os
import typing
import torch
from transformers import PretrainedConfig,PreTrainedModel
from transformers.modeling_utils import shard_checkpoint
from transformers.utils import WEIGHTS_INDEX_NAME


def save_checkpoint_to_hf_format(
        model: typing.Optional[PreTrainedModel,typing.Any],
        output_dir,
        config: typing.Optional[PretrainedConfig] = None,
        max_shard_size="10GB",
):
    if config is not None:
        config.save_pretrained(output_dir)
    state_dict = model.state_dict()
    # Split in shards and save
    shards, index = shard_checkpoint(state_dict,max_shard_size=max_shard_size)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))

    if index is not None:
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
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
            state_dict = torch.load(os.path.join(output_dir, shard_file))
            torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))
    del state_dict
    gc.collect()