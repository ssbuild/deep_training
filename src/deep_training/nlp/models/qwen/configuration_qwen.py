# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import PretrainedConfig


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]


    attribute_map = {
        "n_embd": "hidden_size",
        "n_head": "num_attention_heads",
        "n_positions": "max_position_embeddings",
        "n_layer": "num_hidden_layers",
        "padded_vocab_size": "vocab_size",
    }

    def __init__(
        self,
        vocab_size=151851,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        emb_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=8192,
        scale_attn_weights=True,
        use_cache=True,
        bf16=False,
        fp16=False,
        fp32=False,
        eos_token_id=151643,

        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        use_flash_attn="auto",
        intermediate_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        quantization_bit = 0,
        initializer_weight=False,
        apply_residual_connection_post_layernorm=False,
        pos_emb= "rotary",
        **kwargs,
    ):
        self.eos_token_id = eos_token_id
        super().__init__(
            eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.max_position_embeddings = max_position_embeddings

        self.use_cache = use_cache

        self.bf16 = bf16
        self.fp16 = fp16
        self.fp32 = fp32
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.intermediate_size = intermediate_size
        self.no_bias = no_bias
        self.tie_word_embeddings = tie_word_embeddings

        self.pos_emb = pos_emb
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.quantization_bit = quantization_bit
        self.initializer_weight = initializer_weight
