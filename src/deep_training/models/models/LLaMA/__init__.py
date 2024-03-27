# @Time    : 2023/3/8 0:04
# @Author  : tk
# @FileName: __init__.py
import os
import time
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers import Conv1D, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .configuration import LLaMAConfig
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..transformer import TransformerBase

__all__ = [
    'LLaMAConfig',
    'TransformerLLaMAModel',
    'TransformerLLaMALMHeadModel'
]

MODEL_LOCAL_RANK = 0
MODEL_WORLD_SIZE = 1

def load_pretrain_checkpoint(pretrained_model_name_or_path,config):
    global MODEL_LOCAL_RANK,MODEL_WORLD_SIZE
    local_rank = MODEL_LOCAL_RANK
    world_size = MODEL_WORLD_SIZE
    if local_rank < 0:
        local_rank = 0
    start_time = time.time()

    if pretrained_model_name_or_path is not None:
        checkpoints = sorted(pretrained_model_name_or_path.split(','))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]

        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # third_flag = any([n.find('model.decoder.')!= -1 for n in checkpoint ])
        config: LLaMAConfig
        if config.inference:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            model = LLaMALMHeadModel(config)
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            model = LLaMALMHeadModel(config)
        model.transformer.load_state_dict(checkpoint, strict=False)
    else:
        model = LLaMALMHeadModel(config)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()

        self.n_local_heads = config.n_head
        self.head_dim = config.hidden_size // config.n_head

        self.wq = nn.Linear(
            config.hidden_size,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.hidden_size,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.hidden_size,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.n_head * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.inference = config.inference
        if self.inference:
            self.cache_k = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.inference:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(nn.Module):
    def __init__(self, layer_id: int, config: LLaMAConfig):
        super().__init__()
        self.n_heads = config.n_head
        self.dim = config.hidden_size
        self.head_dim = config.hidden_size // config.n_head
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.hidden_size, hidden_dim=4 * config.hidden_size, multiple_of=config.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



class LLaMAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaMABlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LLaMAModel):
            module.gradient_checkpointing = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):

        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)

        model = load_pretrain_checkpoint(pretrained_model_name_or_path, config)
        return model

class LLaMAModel(LLaMAPreTrainedModel):
    def __init__(self, config: LLaMAConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(LLaMABlock(layer_id, config))

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


        self.freqs_cis = precompute_freqs_cis(
            self.config.hidden_size // self.config.n_head, self.config.max_seq_len * 2
        )
        
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def forward(self, input_ids: torch.Tensor, start_pos: int=0):
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        # if self.config.inference:
        #     h = self.output(h[:, -1]).float()
        # else:
        #     h = self.output(h)
        return h


class LLaMALMHeadModel(LLaMAPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super(LLaMALMHeadModel, self).__init__(config)
        self.transformer = LLaMAModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False,
        )

        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        start_pos: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        transformer_outputs = self.transformer(
            input_ids,start_pos=start_pos
        )
        hidden_states = transformer_outputs

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = (lm_logits,)
        return ((loss,) + output) if loss is not None else output

class TransformerLLaMAModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerLLaMAModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(LLaMAModel, *args, **kwargs))

class TransformerLLaMALMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerLLaMALMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(LLaMALMHeadModel, *args, **kwargs))