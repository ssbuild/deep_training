# coding=utf8
# @Time    : 2023/5/13 17:46
# @Author  : tk
# @FileName: modeling_ilql
import copy
import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Optional, Tuple

from .utils import make_head, hf_get_hidden_size, hf_get_lm_head
from ..chatglm import TransformerChatGlmLMHeadModel
from ..transformer import TransformerForCausalLM, TransformerForSeq2SeqLM

__all__ = [
    'ILQLHeads',
    'AutoModelForCausalLMWithILQLHeads',
    'AutoModelForSeq2SeqLMWithILQLHeads',
]

from ...layers.lora_v2.utils import ModulesToSaveWrapper


def topk_mask(xs: torch.FloatTensor, k: int):
    if k > xs.shape[-1]:
        return xs
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


def batched_index_select(
    x: torch.Tensor , # TensorType["batch", "seq_len", "hidden"],
    idxs: torch.Tensor , # TensorType["batch", "index_len"],
    dim: int,
) -> torch.Tensor:  #TensorType["batch", "index_len", "hidden"]:
    """
    Gather vectors at idxs along dim from x
    """
    idxs = idxs.unsqueeze(-1).expand(idxs.shape[0], idxs.shape[1], x.shape[-1])
    return x.gather(dim=dim, index=idxs)

class ILQLHeads(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            vocab_size: int,
            two_qs: bool,
            alpha: float,
            dtype: type,
            head_size: int,
            up_sampling_score: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.two_qs = two_qs
        self.alpha = alpha
        self.score = make_head(self.hidden_size, head_size, dtype,up_sampling_score=up_sampling_score)

        n_qs = 2 if self.two_qs else 1
        self.q_heads = nn.ModuleList(make_head(self.hidden_size, self.vocab_size, dtype) for _ in range(n_qs))
        self.q_heads_target = nn.ModuleList(copy.deepcopy(q_head) for q_head in self.q_heads)

        for target_q_head in self.q_heads_target:
            target_q_head.requires_grad_(False)

    def forward(
        self,
        hs: torch.Tensor , # TensorType["batch", "seq_len", "hidden"],
        states_ixs: Optional[torch.Tensor] = None, # Optional[TensorType["batch", "states_seq_len"]] = None,
        actions_ixs: Optional[torch.Tensor] = None, #  Optional[TensorType["batch", "actions_seq_len"]] = None,
        **kwargs,
    ) -> Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        torch.Tensor
    ]:  # Tuple[TensorType["batch", "actions_seq_len", "hidden"]],
        # Tuple[TensorType["batch", "actions_seq_len", "hidden"]],
        # TensorType["batch", "states_seq_len", "hidden"]
        if states_ixs is not None:
            states_hs = batched_index_select(hs, states_ixs, 1)
            actions_hs = batched_index_select(hs, actions_ixs, 1)
        else:
            states_hs = actions_hs = hs


        if isinstance(self.q_heads,ModulesToSaveWrapper) and self.q_heads.active_adapter in self.q_heads.modules_to_save:
            q_heads = self.q_heads.modules_to_save[self.q_heads.active_adapter]
        else:
            q_heads = self.q_heads
        qs = tuple(q_head(actions_hs) for q_head in q_heads)
        target_qs = tuple(q_head(actions_hs) for q_head in self.q_heads_target)
        vs = self.score(states_hs)
        return (qs, target_qs, vs)

    def _sync_target_q_heads(self, alpha):
        if isinstance(self.q_heads,
                      ModulesToSaveWrapper) and self.q_heads.active_adapter in self.q_heads.modules_to_save:
            q_heads = self.q_heads.modules_to_save[self.q_heads.active_adapter]
        else:
            q_heads = self.q_heads

        for target_q_head, q_head in zip(self.q_heads_target, q_heads):
            for target_param, copy_param in zip(target_q_head.parameters(), q_head.parameters()):
                target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

    # def sync_target_q_heads(self):
    #     if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") == "3":
    #         with deepspeed.zero.GatheredParameters(list(self.parameters()), modifier_rank=0):
    #             if deepspeed.comm.get_rank() == 0:
    #                 self._sync_target_q_heads(self.alpha)
    #     else:
    #         self._sync_target_q_heads(self.alpha)

    def sync_target_q_heads(self):
        self._sync_target_q_heads(self.alpha)

class AutoModelForCausalLMWithILQLHeads(TransformerForCausalLM):
    """An `AutoModel` class wrapper for `transformers` causal models wtih a language
    modeling head and ILQL heads.

    References:
        [1] Snell et al., "Offline RL for Natural Language Generation with Implicit Language Q Learning",
            https://arxiv.org/abs/2206.11871, 2022
    """


    def __init__(
            self, *args,
            two_qs: bool = True,
            alpha: float = 0.99,
            hidden_size=None, 
            up_sampling_score=False,
            **kwargs,
    ):
        super(AutoModelForCausalLMWithILQLHeads,self).__init__(*args, **kwargs)
        config = self.model.config
        hidden_size = hidden_size or hf_get_hidden_size(config)
        vocab_size = self.config.vocab_size
        dtype = next(hf_get_lm_head(self.model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype,
                                    head_size=config.num_labels,
                                    up_sampling_score=up_sampling_score)

    def forward(
        self,
        *args,
        actions_ixs=None,
        states_ixs=None,
        **kwargs
    ):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.model(*args,**kwargs)
        qs, target_qs, vs = self.ilql_heads(outputs.hidden_states[-1], states_ixs=states_ixs, actions_ixs=actions_ixs)
        return outputs.logits, qs, target_qs, vs, outputs.past_key_values

    def generate_ilql(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp prepossessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])

        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            logits, _, target_qs, vs, past_key_values = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]

            if logit_mask is not None:
                mask = logit_mask[input_ids[:, -1].squeeze().to(logit_mask.device)]
                logits[torch.where(mask)] = -np.inf

            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)
            pi = F.softmax(pi_top_k / temperature, -1)

            input_ids = torch.multinomial(pi, num_samples=1)
            input_ids = (1 - finished) * input_ids + finished * eos_token_id
            finished = (input_ids == eos_token_id).long()

            samples = torch.hstack((samples, input_ids))
            attention_mask = torch.hstack((attention_mask, (input_ids != eos_token_id).long()))
            position_ids = (position_ids[:, -1] + 1).view(-1, 1)

            if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") != "3" and torch.all(finished):
                break

        return samples

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Returns the state dictionary of the model. We add the state dictionary of the ilql heads
    #     to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
    #     """
    #     base_model_state_dict = self.model.state_dict(*args, **kwargs)
    #     ilql_heads_state_dict = self.ilql_heads.state_dict(*args, **kwargs)
    #     for k, v in ilql_heads_state_dict.items():
    #         base_model_state_dict[f"ilql_heads.{k}"] = v
    #     return base_model_state_dict

    # def post_init(self, state_dict):
    #     """
    #     We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
    #     by preprending the key with `ilql_heads.`. This function removes the `ilql_heads.` prefix from the
    #     keys of the value head state dictionary.
    #     """
    #     for k in list(state_dict.keys()):
    #         if "ilql_heads." in k:
    #             state_dict[k.replace("ilql_heads.", "")] = state_dict.pop(k)
    #     self.ilql_heads.load_state_dict(state_dict, strict=False)
    #     del state_dict
    #     gc.collect()





class AutoModelForSeq2SeqLMWithILQLHeads(TransformerForSeq2SeqLM):
    """This is a wrapper around huggingface AutoModelForSeq2Seq with two additional scalar heads"""


    def __init__(
            self, *args,
            two_qs: bool = True,
            alpha: float = 0.99,
            hidden_size=None,
            up_sampling_score=False,
            **kwargs,
    ):
        super().__init__(*args,**kwargs)
        config = self.model.config
        hidden_size = hidden_size or hf_get_hidden_size(config)
        vocab_size = config.vocab_size
        dtype = next(hf_get_lm_head(self.model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype,
                                    head_size=config.num_labels,
                                    up_sampling_score=up_sampling_score)

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Returns the state dictionary of the model. We add the state dictionary of the ilql heads
    #     to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
    #     """
    #     base_model_state_dict = self.model.state_dict(*args, **kwargs)
    #     ilql_heads_state_dict = self.ilql_heads.state_dict(*args, **kwargs)
    #     for k, v in ilql_heads_state_dict.items():
    #         base_model_state_dict[f"ilql_heads.{k}"] = v
    #     return base_model_state_dict
    #
    # def post_init(self, state_dict):
    #     """
    #     We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
    #     by preprending the key with `ilql_heads.`. This function removes the `ilql_heads.` prefix from the
    #     keys of the value head state dictionary.
    #     """
    #     for k in list(state_dict.keys()):
    #         if "ilql_heads." in k:
    #             state_dict[k.replace("ilql_heads.", "")] = state_dict.pop(k)
    #     self.ilql_heads.load_state_dict(state_dict, strict=False)
    #     del state_dict
    #     gc.collect()

    def forward(
        self,
        *args,
        actions_ixs=None,
        states_ixs=None,
        **kwargs,
    ):

        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        out = self.model(**kwargs)
        hs = out.decoder_hidden_states[-1]
        logits = self.model.lm_head(hs)
        qs, target_qs, vs = self.ilql_heads(hs, states_ixs=states_ixs, actions_ixs=actions_ixs)
        encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)
        return logits, qs, target_qs, vs, out.past_key_values, encoder_outputs

    def generate_ilql(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_outputs=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp prepossessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """

        if eos_token_id is None or pad_token_id is None:
            raise ValueError("eos_token_id and pad_token_id must be provided")

        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])
        if decoder_input_ids is None:
            decoder_input_ids = input_ids.new_zeros(input_ids.shape[0], 1)

        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids[:, -1].unsqueeze(-1),
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs,
            )
            logits, _, target_qs, vs, past_key_values, encoder_outputs = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]
            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)
            pi = F.softmax(pi_top_k / temperature, -1)
            next_tokens = torch.multinomial(pi, num_samples=1)
            next_tokens = (1 - finished) * next_tokens + finished * eos_token_id
            finished = (next_tokens == eos_token_id).long() | (next_tokens == pad_token_id).long()
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            samples = decoder_input_ids
            if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") != "3" and torch.all(finished):
                break

        return samples





class ChatglmModelForCausalLMWithILQLHeads(TransformerChatGlmLMHeadModel):
    """An `AutoModel` class wrapper for `transformers` causal models wtih a language
    modeling head and ILQL heads.

    References:
        [1] Snell et al., "Offline RL for Natural Language Generation with Implicit Language Q Learning",
            https://arxiv.org/abs/2206.11871, 2022
    """


    def __init__(
            self, *args,
            two_qs: bool = True,
            alpha: float = 0.99,
            hidden_size=None,
            up_sampling_score=False,
            **kwargs,
    ):
        super(ChatglmModelForCausalLMWithILQLHeads,self).__init__(*args, **kwargs)
        config = self.model.config
        hidden_size = hidden_size or hf_get_hidden_size(config)
        vocab_size = self.config.vocab_size
        dtype = next(hf_get_lm_head(self.model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype,
                                    head_size=config.num_labels,
                                    up_sampling_score=up_sampling_score)

    def forward(
        self,
        *args,
        actions_ixs=None,
        states_ixs=None,
        **kwargs
    ):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.model(*args,**kwargs)
        qs, target_qs, vs = self.ilql_heads(outputs.hidden_states[-1], states_ixs=states_ixs, actions_ixs=actions_ixs)
        return outputs.logits, qs, target_qs, vs, outputs.past_key_values

    def generate_ilql(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp prepossessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])

        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            logits, _, target_qs, vs, past_key_values = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]

            if logit_mask is not None:
                mask = logit_mask[input_ids[:, -1].squeeze().to(logit_mask.device)]
                logits[torch.where(mask)] = -np.inf

            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)
            pi = F.softmax(pi_top_k / temperature, -1)

            input_ids = torch.multinomial(pi, num_samples=1)
            input_ids = (1 - finished) * input_ids + finished * eos_token_id
            finished = (input_ids == eos_token_id).long()

            samples = torch.hstack((samples, input_ids))
            attention_mask = torch.hstack((attention_mask, (input_ids != eos_token_id).long()))
            position_ids = (position_ids[:, -1] + 1).view(-1, 1)

            if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") != "3" and torch.all(finished):
                break

        return samples

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Returns the state dictionary of the model. We add the state dictionary of the ilql heads
    #     to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
    #     """
    #     base_model_state_dict = self.model.state_dict(*args, **kwargs)
    #     ilql_heads_state_dict = self.ilql_heads.state_dict(*args, **kwargs)
    #     for k, v in ilql_heads_state_dict.items():
    #         base_model_state_dict[f"ilql_heads.{k}"] = v
    #     return base_model_state_dict

    # def post_init(self, state_dict):
    #     """
    #     We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
    #     by preprending the key with `ilql_heads.`. This function removes the `ilql_heads.` prefix from the
    #     keys of the value head state dictionary.
    #     """
    #     for k in list(state_dict.keys()):
    #         if "ilql_heads." in k:
    #             state_dict[k.replace("ilql_heads.", "")] = state_dict.pop(k)
    #     self.ilql_heads.load_state_dict(state_dict, strict=False)
    #     del state_dict
    #     gc.collect()