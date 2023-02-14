# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 16:31
from typing import Optional, Union, Iterable, Sequence
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t


def DataLoaderDistributed(dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
    # num_replicas = dist.get_world_size()
    # if num_replicas > 1:
    #     generator = DistributedSampler()
    # else:
    #     generator = None
    return DataLoader(dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=shuffle,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    timeout=timeout,
                    worker_init_fn=worker_init_fn,
                    multiprocessing_context=multiprocessing_context,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    pin_memory_device=pin_memory_device,
                    generator=generator)