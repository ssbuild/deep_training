# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 14:15
from itertools import repeat
from .logging import get_logger
import time
from dataclasses import dataclass
from typing import Dict, MutableMapping, Union, Tuple, Mapping, Iterable

import numpy as np
from transformers import PretrainedConfig
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
logger = get_logger(__name__)

@dataclass
class RLElement:
    """
    Batch element for RL model
    """

    state: Iterable[str] = None  # Context/prompts
    action: torch.Tensor = None #TensorType["N"] = None  # Tokens generated by model given prompts
    reward: float = None  # Reward obtained for that generation

@dataclass
class PPORLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over the response tokens generated
                    by the policy network (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens.
    :type logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    query_tensor: torch.Tensor
    response_tensor: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


@dataclass
class PromptBatch:
    """
    Batched PromptElement

    :param text: An iterable of prompt texts.
    :type text: Iterable[str]

    :param tokens: A long tensor batch of prompt tokens.
    :type tokens: torch.Tensor
    """

    text: Iterable[str]
    tokens: torch.Tensor  # TensorType["batch_size", "num_tokens"]


@dataclass
class PPORLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    query_tensors: torch.Tensor  # TensorType["batch_size", "query_size"]
    response_tensors: torch.Tensor  # TensorType["batch_size", "response_size"]
    logprobs: torch.Tensor  # TensorType["batch_size", "response_size"]
    values: torch.Tensor  # TensorType["batch_size", "response_size"]
    rewards: torch.Tensor  # TensorType["batch_size", "response_size"]


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

    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)

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


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)

def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))

def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs:
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def _gpu_gather(tensor,world_size):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(world_size)]
        dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


_cpu_gather = _gpu_gather

def pad_across_processes(tensor,world_size, dim=0, pad_index=0, pad_first=False):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so they
    can safely be gathered.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.
        dim (`int`, *optional*, defaults to 0):
            The dimension on which to pad.
        pad_index (`int`, *optional*, defaults to 0):
            The value with which to pad.
        pad_first (`bool`, *optional*, defaults to `False`):
            Whether to pad at the beginning or the end.
    """

    def _pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
        if dim >= len(tensor.shape):
            return tensor

        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = _gpu_gather(size,world_size).cpu()
        # Then pad to the maximum size
        max_size = max(s[dim] for s in sizes)
        if max_size == tensor.shape[dim]:
            return tensor

        old_size = tensor.shape
        new_size = list(old_size)
        new_size[dim] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        if pad_first:
            indices = tuple(
                slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))
            )
        else:
            indices = tuple(slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size)))
        new_tensor[indices] = tensor
        return new_tensor

    return recursively_apply(
        _pad_across_processes, tensor, error_on_other_type=True, dim=dim, pad_index=pad_index, pad_first=pad_first
    )



def infinite_dataloader(dataloader: Iterable) -> Iterable:
    """
    Returns a cyclic infinite dataloader from a finite dataloader
    """
    for _ in repeat(dataloader):
        yield from dataloader