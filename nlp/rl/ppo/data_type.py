# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 14:13
from dataclasses import dataclass
from typing import Iterable

import torch


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
    tokens:  torch.Tensor #TensorType["batch_size", "num_tokens"]
    
    



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

    query_tensors: torch.Tensor #TensorType["batch_size", "query_size"]
    response_tensors:  torch.Tensor #TensorType["batch_size", "response_size"]
    logprobs:  torch.Tensor #TensorType["batch_size", "response_size"]
    values:  torch.Tensor #TensorType["batch_size", "response_size"]
    rewards:  torch.Tensor #TensorType["batch_size", "response_size"]