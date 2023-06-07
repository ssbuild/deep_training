# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" RWKV configuration"""
from transformers import PretrainedConfig

# from ...configuration_utils import PretrainedConfig
# from ...utils import logging
#
#
# logger = logging.get_logger(__name__)

RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "RWKV/rwkv-4-169m-pile": "https://huggingface.co/RWKV/rwkv-4-169m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-430m-pile": "https://huggingface.co/RWKV/rwkv-4-430m-pile/resolve/main/config.json",
    "RWKV/rwkv-4-1b5-pile": "https://huggingface.co/RWKV/rwkv-4-1b5-pile/resolve/main/config.json",
    "RWKV/rwkv-4-3b-pile": "https://huggingface.co/RWKV/rwkv-4-3b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-7b-pile": "https://huggingface.co/RWKV/rwkv-4-7b-pile/resolve/main/config.json",
    "RWKV/rwkv-4-14b-pile": "https://huggingface.co/RWKV/rwkv-4-14b-pile/resolve/main/config.json",
    "RWKV/rwkv-raven-1b5": "https://huggingface.co/RWKV/rwkv-raven-1b5/resolve/main/config.json",
    "RWKV/rwkv-raven-3b": "https://huggingface.co/RWKV/rwkv-raven-3b/resolve/main/config.json",
    "RWKV/rwkv-raven-7b": "https://huggingface.co/RWKV/rwkv-raven-7b/resolve/main/config.json",
    "RWKV/rwkv-raven-14b": "https://huggingface.co/RWKV/rwkv-raven-14b/resolve/main/config.json",
}




class RwkvConfig(PretrainedConfig):
    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length","attention_hidden_size": "n_embd",
                     "context_length": "ctx_len","hidden_size": "n_embd","intermediate_size": "dim_ffn"
                     }

    def __init__(
            self,
            vocab_size=50277,
            ctx_len=1024,
            n_embd = 768,
            n_layers=12,
            layer_norm_epsilon=1e-5,
            bos_token_id=0,
            eos_token_id=0,
            rescale_every=6,
            tie_word_embeddings=False,
            use_cache=True,
            dim_att = 0,
            dim_ffn= 0,
            pos_emb_size=0,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_layers = n_layers

        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.dim_att = dim_att
        self.dim_ffn = dim_ffn
        # self.tiny_att_layer = tiny_att_layer
        self.pos_emb_size = pos_emb_size


        if dim_att == 0:
            self.dim_att = self.n_embd
        if dim_ffn == 0:
            self.dim_ffn = self.n_embd * 4
        # if tiny_att_layer <= 0 or not tiny_att_layer:
        #     self.tiny_att_layer = -1


        super().__init__(
            tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )

