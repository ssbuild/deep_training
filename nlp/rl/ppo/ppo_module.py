# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 11:05
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from .configuration import PPOConfig
from ..utils import logprobs_of_labels, get_tensor_stats, flatten_dict, whiten
from .data_define import PPORLBatch
from ...models.rl.utils import CausalLMOutputWithValue



class PPOLLMAbstract:
    def forward_llm_value_and_logits(self,input_ids,**kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids,**kwargs)
        logits = outputs.logits
        values_pred = outputs.value
        return (logits,values_pred)

class PPOSEQ2SEQAbstract:
    def forward_seq2seq_value_and_logits(self,
                                         input_ids,attention_mask,
                                         decoder_input_ids,decoder_attention_mask,
                                         **kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             decoder_input_ids=decoder_input_ids,
                                             decoder_attention_mask=decoder_attention_mask,
                                             **kwargs)
        logits = outputs.logits
        values_pred = outputs.value
        return (logits, values_pred)

class PPOPrefixLMAbstract:
    def forward_prefix_value_and_logits(self,input_ids,**kwargs):
        outputs = self.forward_logits_values(input_ids=input_ids, **kwargs)
        logits = outputs.logits
        values_pred = outputs.value
        return (logits, values_pred)


class PPOModelLoss(nn.Module, PPOLLMAbstract, PPOSEQ2SEQAbstract,PPOPrefixLMAbstract):
    def forward_ppo_loss(self,batch: PPORLBatch, device):
        """Forward pass & loss
          Args:
              batch: Previous batch of episodes
        """
        query_tensors = batch.query_tensors.to(device)
        response_tensors = batch.response_tensors.to(device)
        old_logprobs = batch.logprobs.to(device)
        old_values = batch.values.to(device)
        old_rewards = batch.rewards.to(device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, response_length)
        if self.ppo_config.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = input_ids.ne(self.config.pad_token_id).long().to(device)
            decoder_attention_mask = (
                decoder_input_ids.ne(self.config.pad_token_id).long().to(device)
            )
            decoder_attention_mask[:, 0] = 1

            # Forward pass
            logits,values_pred = self.forward_seq2seq_value_and_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )

            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = decoder_input_ids.ne(self.config.pad_token_id).long().to(device)
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start:end],
            )
        elif self.ppo_config.model_arch_type == "prefixlm":
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = tokens.not_equal(self.config.pad_token_id).long().to(tokens.device)
            logits, values_pred = self.forward_prefix_value_and_logits(input_ids=tokens,return_dict=True)
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
            )
        else:
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = tokens.not_equal(self.config.pad_token_id).long().to(tokens.device)
            logits,values_pred = self.forward_llm_value_and_logits(input_ids=tokens,
                                                                   attention_mask=attention_mask,
                                                                   return_dict=True)
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
            )

        loss, stats = self.loss_fn(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        return {
            'loss': loss,
            'stats': stats
        }

    def get_advantages_and_returns(
            self,
            values, # : TensorType["batch_size", "response_size"]
            rewards, #: TensorType["batch_size", "response_size"]
            response_length: int,
            use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.ppo_config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_config.gamma * self.ppo_config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss_fn(
            self,
            logprobs, # : TensorType["batch_size", "response_size"]
            values, # : TensorType["batch_size", "response_size"]
            old_logprobs, # : TensorType["batch_size", "response_size"]
            old_values, # : TensorType["batch_size", "response_size"]
            advantages, # : TensorType["batch_size", "response_size"]
            returns, # : TensorType["batch_size", "response_size"]
            mask # : TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.ppo_config.cliprange_value,
            old_values + self.ppo_config.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.ppo_config.cliprange,
            1.0 + self.ppo_config.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.ppo_config.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=n / mask.numel(),
        )
        return loss, flatten_dict(stats)
