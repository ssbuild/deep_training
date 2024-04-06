# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert a RWKV checkpoint from BlinkDL to the Hugging Face format."""


import argparse
import gc
import json
import os
import re
import shutil
from typing import Union

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint
from deep_training.nlp.models.rwkv4.configuration_rwkv import RwkvConfig


NUM_HIDDEN_LAYERS_MAPPING = {
    "0.1B": 12,
    "0.4B": 24,
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

HIDEN_SIZE_MAPPING = {
    "0.1B": 768,
    "0.4B": 1024,
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}


def convert_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)
        if name != "head.weight":
            name = "rwkv." + name
        state_dict[name] = weight
    return state_dict


def convert_rmkv_checkpoint_to_hf_format(
        repo_id, checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None,ctx_len=1024,
        vocab_size=None,
        max_shard_size: Union[int, str] = "10GB",
        is_world=False
):
    # 1. If possible, build the tokenizer.
    # if tokenizer_file is None:
    #     print("No `--tokenizer_file` provided, we will use the default tokenizer.")
    #     vocab_size = 50277
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # else:
    #     tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    #     vocab_size = len(tokenizer)
    # tokenizer.save_pretrained(output_dir)

    if tokenizer_file is not None:
        if not is_world:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            if vocab_size is None:
                vocab_size = len(tokenizer)
            tokenizer.save_pretrained(output_dir)
        else:
            shutil.copy(tokenizer_file,os.path.join(output_dir,os.path.basename(tokenizer_file)))

    # 2. Build the config
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    if size is None:
        # Try to infer size from the checkpoint name
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError("Could not infer the size, please provide it with the `--size` argument.")
    if size not in possible_sizes:
        raise ValueError(f"`size` should be one of {possible_sizes}, got {size}.")

    config = RwkvConfig(
        vocab_size=vocab_size,
        n_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        n_embd=HIDEN_SIZE_MAPPING[size],
        ctx_len = ctx_len,
    )
    config.save_pretrained(output_dir)

    # 3. Download model file then convert state_dict
    if os.path.exists(checkpoint_file) and os.path.isfile(checkpoint_file):
        model_file = checkpoint_file
    else:
        model_file = hf_hub_download(repo_id, checkpoint_file)
    state_dict = torch.load(model_file, map_location="cpu")
    state_dict = convert_state_dict(state_dict)

    # 4. Split in shards and save
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

    if push_to_hub:
        if model_name is None:
            raise ValueError("Please provide a `model_name` to push the model to the Hub.")
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        model.push_to_hub(model_name, max_shard_size="2GB")
        if tokenizer is not None:
            tokenizer.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--repo_id", default=None, type=str, help="Repo ID from which to pull the checkpoint."
    )
    parser.add_argument(
        "--checkpoint_file", default=None, type=str, required=True, help="Name of the checkpoint file in the repo."
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True, help="Where to save the converted model."
    )
    parser.add_argument(
        "--tokenizer_file",
        default=None,
        type=str,
        help="Path to the tokenizer file to use (if not provided, only the model is converted).",
    )
    parser.add_argument(
        "--size",
        default=None,
        type=str,
        help="Size of the model. Will be inferred from the `checkpoint_file` if not passed.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to the Hub the converted model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the pushed model on the Hub, including the username / organization.",
    )

    parser.add_argument(
        "--ctx_len",
        default=1024,
        type=int,
        help="Size of the model. Will be inferred from the `checkpoint_file` if not passed.",
    )

    args = parser.parse_args()
    convert_rmkv_checkpoint_to_hf_format(
        args.repo_id,
        args.checkpoint_file,
        args.output_dir,
        size=args.size,
        tokenizer_file=args.tokenizer_file,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
        ctx_len=args.ctx_len,
    )
