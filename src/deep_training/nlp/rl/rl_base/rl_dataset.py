# coding=utf8
# @Time    : 2023/5/14 23:14
# @Author  : tk
# @FileName: rl_dataset
from abc import abstractmethod
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Iterable
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding

from ..utils.logging import get_logger

logger = get_logger(__name__)

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

    def __getitem__(self, index: int):
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