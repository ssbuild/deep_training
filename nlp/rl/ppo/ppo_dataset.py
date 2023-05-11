# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:34

import json
import os
from abc import abstractmethod
from dataclasses import is_dataclass
from time import time
from typing import List, Callable, Tuple
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import Any, Callable, Dict, Iterable
from torch.utils.data import DataLoader, Dataset
from .utils import logprobs_of_labels, Clock, gather_dict, RunningMoments, pad_across_processes, _gpu_gather, \
    PPORLElement, RLElement, PPORLBatch
from .utils import logging, logger


class BaseRolloutStore(Dataset):
    def __init__(self, capacity=-1):
        self.history: Iterable[Any] = None
        self.capacity = capacity

    @abstractmethod
    def push(self, exps: Iterable[Any]):
        """
        Push experiences to rollout storage
        """
        pass

    def __getitem__(self, index: int) -> RLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    @abstractmethod
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a dataloader for the rollout store

        :param prep_fn: Applied to RLElement after collation (typically tokenizer)
        :type prep_fn: Callable
        """
        pass


class MiniBatchIterator:
    """
    A custom iterator for generating mini-batches from a PyTorch DataLoader.
    """

    def __init__(self, data_loader, mb_size, num_mb):
        """
        Initializes the MiniBatchIterator.

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to generate mini-batches from.
            mb_size (int): The size of each mini-batch.
            num_mb (int): The number of mini-batches to generate for each iteration.
        """
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.mb_size = mb_size
        self.num_mb = num_mb

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.data_loader_iter)
        minibatches = []

        for mbi in range(self.num_mb):
            sliced_data = {}
            batch_dict = batch
            if is_dataclass(batch):
                batch_dict = batch.__dict__
            for key, value in batch_dict.items():
                start_idx = mbi * self.mb_size
                end_idx = (mbi + 1) * self.mb_size
                sliced_data[key] = value[start_idx:end_idx]

                if len(sliced_data[key]) == 0:
                    logger.warning(
                        "WARNING: MiniBatchIterator generated a minibatch with 0 elements. "
                        "This may be due to the wrong mb_size and/or num_mb or the last batch"
                        "in the dataset being smaller."
                    )
                    sliced_data.pop(key)
                    break
                elif len(sliced_data[key]) < self.mb_size:
                    logger.warning(
                        "WARNING: MiniBatchIterator generated a minibatch with fewer elements than mb_size. "
                        "This may be due to the wrong mb_size and/or num_mb or the last batch in the dataset "
                        "being smaller."
                    )
            if not sliced_data:
                break

            if isinstance(batch, BatchEncoding):
                minibatch = BatchEncoding(sliced_data)
            elif is_dataclass(batch):
                minibatch = batch.__class__(**sliced_data)
            # else:
            #     minibatch = sliced_data

            minibatches.append(minibatch)

        if not minibatches:
            raise StopIteration

        return minibatches




class PPORolloutStore(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self,
        batch_size: int,
        shuffle: bool,
        **kwargs,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            if self.padding_side == "right":
                # Right padding of already right-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                )
            else:
                # Left padding of already left-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1)

            return PPORLBatch(
                query_tensors,
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)


