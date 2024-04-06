# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/19 14:36
from typing import Dict, Tuple,Union,List
import torch
from deep_training.nlp.losses.loss_dpo import dpo_loss
from torch import nn

class DpoModule:
    def set_ref_model(self, ref_model):
        self.ref_model = ref_model

    def forward_logits(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], n) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        labels = batch.pop('labels')
        outputs = model(**batch, return_dict=True)
        batch["labels"] = labels
        all_logits = outputs.logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, labels, average_log_prob=False)
        chosen_logps = all_logps[:n]
        rejected_logps = all_logps[n:]
        return chosen_logps, rejected_logps

    def compute_loss(self, *args, **batch) -> tuple:
        if self.training:
            inputs = {}
            ks = set(k for k in batch if k.find('2') == -1)
            for k in ks:
                inputs[k] = torch.cat((batch[k], batch[k + '2']), dim=0)
        else:
            inputs = batch
        n = batch['input_ids'].shape[0]
        chosen_logps, rejected_logps = self.forward_logits(model=self, batch=inputs, n=n)
        returns = tuple()
        if self.training:
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = self.forward_logits(model=self.ref_model.backbone.model,
                                                                           batch=inputs, n=n)
            losses, chosen_rewards, rejected_rewards = dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps,
                                                                ref_rejected_logps, beta=self.beta,
                                                                reference_free=self.ref_free)
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            loss_dict = {
                "loss": losses.mean(),
                "chosen_rewards": chosen_rewards.mean(),
                "rejected_rewards": rejected_rewards.mean(),
                "reward_accuracies": reward_accuracies.mean(),
            }
            returns += (loss_dict, chosen_rewards, rejected_rewards)
        else:
            returns += (chosen_logps, rejected_logps)
        return returns


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.Tensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)