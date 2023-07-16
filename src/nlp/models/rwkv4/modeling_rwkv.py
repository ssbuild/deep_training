# -*- coding: utf-8 -*-
# @Time:  12:39
# @Author: tk
# @File：rwkv-4
import math
import os
import typing
from typing import Optional, List, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from .configuration_rwkv import RwkvConfig
from ..transformer_base import TransformerBase

__T_MAX__: int = 1024
__WKV_CUDA__ : typing.Optional = None

__all__ = [
    'RwkvCausalLMOutput',
    'RwkvModel',
    'RwkvCausalLMOutput',
    'RwkvForCausalLM',
    'TransformerRWKV4LMHeadModel',
]

def set_model_profile(RWKV_T_MAX,RWKV_FLOAT_MODE='32'):
    from torch.utils import cpp_extension
    global __T_MAX__,__WKV_CUDA__
    # torch._C._jit_set_profiling_executor(True)
    # torch._C._jit_set_profiling_mode(True)
    __T_MAX__ = RWKV_T_MAX  # TAKES LOTS OF VRAM!
    cur_path = os.path.dirname(__file__)
    if RWKV_FLOAT_MODE == "bf16":
        __WKV_CUDA__ = cpp_extension.load(name=f"deep_wkv_{__T_MAX__}_bf16", sources=[os.path.join(cur_path,_) for _ in ["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"]],
                        verbose=True,
                        extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math",
                                           "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={__T_MAX__}"])
    elif RWKV_FLOAT_MODE in ['16','32']:
        __WKV_CUDA__ = cpp_extension.load(name=f"deep_wkv_{__T_MAX__}", sources=[os.path.join(cur_path,_)  for _ in ["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"]], verbose=True,
                        extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3",
                                           "--extra-device-vectorization", f"-DTmax={__T_MAX__}"])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v, st,return_state):
        B, T, C = k.size()
        ctx.B = B
        ctx.T = T
        ctx.C = C

        assert T <= __T_MAX__
        assert B * C % min(C, 32) == 0
        input_dtype = k.dtype
        ctx.input_dtype = input_dtype

        if st is None:
            s = torch.zeros(B,C,3,
                dtype=torch.float32,
                device=k.device,
                # memory_format=torch.contiguous_format,
            )
            s[:, :, 2] -= 1e38
        else:
            s = torch.cat([_.unsqueeze(2) for _ in st], dim=2).contiguous()

        if input_dtype == torch.float16:
            # inference
            if w.dtype == torch.float16:
                w = w.float()
                u = u.float()

            k = k.float()
            v = v.float()

        if s.dtype != torch.float32:
            s = s.float()

        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        __WKV_CUDA__.forward(w, u, k, v, s, y)
        ctx.save_for_backward(w, u, k, v, y)

        if return_state and s is not None:
            st = [_.squeeze(2) for _ in torch.chunk(s, 3, dim=2)]
        return y.to(input_dtype), st

    @staticmethod
    def backward(ctx, gy, gstate=None):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        input_dtype = ctx.input_dtype
        assert T <= __T_MAX__
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors
        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16 if input_dtype == torch.bfloat16 else torch.float32)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)

        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        if input_dtype == torch.float16:
            gy = gy.float()
        __WKV_CUDA__.backward(w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (gw.to(input_dtype), gu.to(input_dtype), gk.to(input_dtype), gv.to(input_dtype), None, None)


def RUN_CUDA(w, u, k, v,s,return_state):
    if __WKV_CUDA__ is None:
        return rwkv_linear_attention_cpu(w, u, k, v,s,return_state)
    return WKV.apply(w, u, k, v, s,return_state)



def rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=None,return_state=False):
    # For CPU fallback. Will be slower and probably take more memory than the custom CUDA kernel if not executed
    # within a torch.no_grad.
    _, seq_length, _ = key.size()
    output = torch.zeros_like(key)

    if state is None:
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
    else:
        num_state, den_state, max_state = state
    # For numerical stability
    #    real_numerator_state = num_state * torch.exp(max_state)
    #    real_denominator_state = den_state * torch.exp(max_state)

    time_decay = -torch.exp(time_decay)

    for current_index in range(seq_length):
        current_key = key[:, current_index].float()
        current_value = value[:, current_index]

        # wkv computation at time t
        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)

        # Update state for next iteration
        max_for_state = torch.maximum(max_state + time_decay, current_key)
        e1 = torch.exp(max_state + time_decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    if return_state and state is not None:
        state = [num_state, den_state, max_state]

    return output, state


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RwkvSelfAttention(nn.Module):
    def __init__(self, config: RwkvConfig, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        self.time_decay = nn.Parameter(torch.empty(config.dim_att))
        self.time_first = nn.Parameter(torch.empty(config.dim_att))
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, config.dim_att))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, config.dim_att))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, config.dim_att))

        # with torch.no_grad():  # fancy init
        #     ratio_0_to_1 = layer_id / (config.n_layers - 1)  # 0 to 1
        #     ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layers)  # 1 to ~0
        #     ddd = torch.ones(1, 1, config.n_embd)
        #     for i in range(config.n_embd):
        #         ddd[0, 0, i] = i / config.n_embd
        #
        #     # fancy time_decay
        #     decay_speed = torch.ones(config.dim_att)
        #     for h in range(config.dim_att):
        #         decay_speed[h] = -5 + 8 * (h / (config.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        #     self.time_decay = nn.Parameter(decay_speed)
        #
        #     # fancy time_first
        #     zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(config.dim_att)]) * 0.5
        #     self.time_first = nn.Parameter(torch.ones(config.dim_att) * math.log(0.3) + zigzag)
        #
        #     # fancy time_mix
        #     self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        #     self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        #     self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.value = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False)
        self.output = nn.Linear(config.dim_att, config.n_embd, bias=False)

    def jit_func(self, x, state=None):
        # Mix x with the previous timestep to produce xk, xv, xr
        if x.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(x)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]

        xx = shifted
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        if state is not None:
            state[1][:, :, self.layer_id] = x[:, -1]
        return sr, k, v, state

    def forward(self, x, state=None,use_cache=None):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v, state = self.jit_func(x,state=state)

        layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None
        rwkv,layer_state = RUN_CUDA(self.time_decay, self.time_first, k, v, layer_state,return_state=use_cache)
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]
        return self.output(sr * rwkv),state


class RwkvFeedForward(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # with torch.no_grad():  # fancy init of time_mix
        #     ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layers)  # 1 to ~0
        #     ddd = torch.ones(1, 1, config.n_embd)
        #     for i in range(config.n_embd):
        #         ddd[0, 0, i] = i / config.n_embd
        #     self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        #     self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, config.n_embd))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, config.n_embd))

        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)


    def forward(self, x, state=None):
        if x.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(x)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        xx = shifted
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        if state is not None:
            state[0][:, :, self.layer_id] = x[:, -1]
        return torch.sigmoid(self.receptance(xr)) * kv,state


class MishGLU(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layers)

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
            self.bb = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
            self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)


    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class RwkvBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config: RwkvConfig = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
            if config.pos_emb_size > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1, config.pos_emb_size, config.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((config.pos_emb_size, 1, config.n_embd)))


        self.att = RwkvSelfAttention(config, layer_id)
        # if 'g' in os.environ["RWKV_MY_TESTING"]:
        #     self.ffn = MishGLU(config, layer_id)
        self.ffn = RwkvFeedForward(config, layer_id)


    def forward(self, x, x_emb=None,state=None, use_cache=False, output_attentions=False):
        config = self.config
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if config.pos_emb_size > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]
                x = x + pos_emb


        attention,state = self.att(self.ln1(x),state=state,use_cache=use_cache)
        x = x + attention
        feed_forward, state = self.ffn(self.ln2(x),state=state)
        x = x + feed_forward

        outputs = (x, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        return outputs




class RwkvPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RwkvConfig
    base_model_prefix = "rwkv"
    _no_split_modules = ["RwkvBlock"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, RwkvSelfAttention):
            layer_id = module.layer_id
            n_layers = module.config.n_layers
            n_embd = module.config.n_embd
            attention_hidden_size = module.config.dim_att

            ratio_0_to_1 = layer_id / (n_layers - 1 + 1e-12)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / n_embd for i in range(n_embd)],
                dtype=module.time_mix_k.dtype,
                device=module.time_mix_k.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            zigzag = (
                torch.tensor(
                    [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                    dtype=module.time_first.dtype,
                    device=module.time_first.device,
                )
                * 0.5
            )

            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(module.time_first * math.log(0.3) + zigzag)

                module.time_mix_k.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_v.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_r.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        elif isinstance(module, RwkvFeedForward):
            layer_id = module.layer_id
            n_layers = module.config.n_layers
            n_embd = module.config.n_embd

            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / n_embd for i in range(n_embd)],
                dtype=module.time_mix_k.dtype,
                device=module.time_mix_k.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_mix_k.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_r.data = torch.pow(time_weight, ratio_1_to_almost0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RwkvModel):
            module.gradient_checkpointing = value





@dataclass
class RwkvOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, n_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RwkvCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, n_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RwkvModel(RwkvPreTrainedModel):
    def __init__(self, config: RwkvConfig):
        super().__init__(config)
        self.config: RwkvConfig = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RwkvBlock(config, i) for i in range(config.n_layers)])
        self.ln_out = nn.LayerNorm(config.n_embd)

        self._is_weight_rescaled = False
        # Initialize weights and apply final processing
        self.post_init()




    def _rescale_weight(self):
        with torch.no_grad():
            for block_id, block in enumerate(self.blocks):
                if self.training:
                    block.att.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    block.ffn.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                else:
                    # Deal with quantization statistics
                    if hasattr(block.att.output.weight, "SCB"):
                        block.att.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                    elif hasattr(block.att.output.weight, "quant_state"):
                        block.att.output.weight.quant_state[0].div_(
                            2 ** int(block_id // self.config.rescale_every)
                        )
                        block.ffn.value.weight.quant_state[0].div_(
                            2 ** int(block_id // self.config.rescale_every)
                        )
                    else:
                        block.att.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.div_(2 ** int(block_id // self.config.rescale_every))


    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,  # noqa
                inputs_embeds: Optional[torch.FloatTensor] = None,
                state: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                return_state_only = False,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        config = self.config

        if not self.training:
            if not self._is_weight_rescaled:
                self._rescale_weight()
                self._is_weight_rescaled = True


        B, T = input_ids.size()
        assert T <= config.ctx_len, "Cannot forward, model ctx_len is exhausted."

        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)


        if use_cache and state is None:
            shape = (inputs_embeds.size(0), self.config.n_embd, self.config.n_layers)
            state = [
                torch.zeros(
                    *shape, dtype=inputs_embeds.dtype if i <= 1 else torch.float32, device=inputs_embeds.device
                )
                for i in range(5)
            ]
            state[4] -= 1e30

        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None


        for i,block in enumerate(self.blocks):
            hidden_states, state, attentions = block(hidden_states,state=state, use_cache=use_cache, output_attentions=output_attentions)

            if not self.training and self.config.rescale_every >0:
                if (i + 1) % self.config.rescale_every == 0:
                    hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        if return_state_only:
            if not return_dict:
                return (state,)
            else:
                return RwkvOutput(
                    state=state,
                )


        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(x for x in [hidden_states, state, all_hidden_states, all_self_attentions] if x is not None)

        return RwkvOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # def get_model_parameters(self):
    #     config = self.config
    #     if config.layerwise_lr > 0:
    #         lr_1x = set()
    #         lr_2x = set()
    #         lr_3x = set()
    #         for n, p in self.named_parameters():
    #             if "time_mix" in n:
    #                 if config.pile_stage == 2:
    #                     lr_2x.add(n)
    #                 else:
    #                     lr_1x.add(n)
    #             elif "time_decay" in n:
    #                 if config.pile_stage == 2:
    #                     lr_3x.add(n)
    #                 else:
    #                     lr_2x.add(n)
    #             elif "time_first" in n:
    #                 lr_3x.add(n)
    #             else:
    #                 lr_1x.add(n)
    #         lr_1x = sorted(list(lr_1x))
    #         lr_2x = sorted(list(lr_2x))
    #         lr_3x = sorted(list(lr_3x))
    #         # print('1x', lr_1x)
    #         # print('2x', lr_2x)
    #         # print('3x', lr_3x)
    #         param_dict = {n: p for n, p in self.named_parameters()}
    #         if config.pile_stage == 2:
    #             optim_groups = [
    #                 {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
    #                 {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
    #                 # test: 2e-3 / config.lr_init},
    #                 {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
    #                 # test: 3e-3 / config.lr_init},
    #             ]
    #         else:
    #             optim_groups = [
    #                 {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
    #                 {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
    #                 {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
    #             ]
    #     else:
    #         optim_groups = [
    #             {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
    #         ]
    #
    #     return optim_groups


class RwkvForCausalLM(RwkvPreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config: RwkvConfig):
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.rwkv.emb

    def set_input_embeddings(self, value):
        self.rwkv.emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_state_only=False,
    ) -> Union[Tuple, RwkvCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_state_only=return_state_only,
        )

        if return_state_only:
            if not return_dict:
                return (state,)
            else:
                return RwkvCausalLMOutput(
                    state=rwkv_outputs.state,
                )


        hidden_states = rwkv_outputs[0]



        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )



class TransformerRWKV4LMHeadModel(TransformerBase):
    def __init__(self, *args,**kwargs):
        super(TransformerRWKV4LMHeadModel, self).__init__(*args,**kwargs)
        self.set_model(self.from_pretrained(RwkvForCausalLM, *args, **kwargs))