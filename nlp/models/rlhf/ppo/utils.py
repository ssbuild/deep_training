# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 14:15
import time
from typing import Dict, MutableMapping, Union, Tuple

import numpy as np
from transformers import PretrainedConfig
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F




def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def gather_dict(obj: Dict,
                # grad_state: accelerate.state.GradientState = None
                ):
    """
    Gather and concatenates key-values from a dictionary, optionally
    trimming them if some of them were out of dataloader's padding
    """
    if not torch.distributed.is_initialized():
        return obj

    objs = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(objs, obj)

    acc, *objs = objs
    for obj in objs:
        for k in obj:
            acc[k].extend(obj[k])

    # if grad_state:
    #     if grad_state.end_of_dataloader and grad_state.remainder > 0:
    #         for k in acc:
    #             acc[k] = acc[k][: grad_state.remainder]

    return acc


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    if xs.numel() == 0:
        return dict(mean=0, min=0, max=0, std=0)

    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


class Clock:
    """
    Helper object for keeping track of time for computations.
    """

    def __init__(self):
        self.start = time.time()
        self.total_time = 0
        self.total_samples = 0

    def tick(self, samples: int = 0) -> float:
        """
        Returns time (s) since last call to tick(). Also records samples processed since last call.

        :param samples: number of samples that have been processed since last call
        """
        end = time.time()
        delta = end - self.start
        self.start = end

        if samples != 0:
            self.total_time += delta
            self.total_samples += samples

        return delta

    def get_stat(self, n_samp: int = 1000, reset: bool = False):
        """
        Returns average time (s) per n_samp samples processed

        :param reset: Reset counts?
        """
        sec_per_samp = self.total_time / self.total_samples

        if reset:
            self.total_samples = 0
            self.total_time = 0

        return sec_per_samp * n_samp




class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()
