# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:34

import json
import os
from abc import abstractmethod
from dataclasses import is_dataclass
from time import time
from typing import List, Callable, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import Any, Callable, Dict, Iterable
from torch.utils.data import DataLoader, Dataset
from .data_define import PPORLElement, RLElement, PPORLBatch,logger
from ..rl_base.rl_dataset import BaseRolloutStore



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


