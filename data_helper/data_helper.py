# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02
import json
import os
import typing
import torch
import numpy as np
from fastdatasets.record import RECORD
from fastdatasets.leveldb import LEVELDB
from fastdatasets.lmdb import LMDB
from fastdatasets.utils.numpyadapter import NumpyReaderAdapter,E_file_backend
from torch.utils.data import DataLoader
from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset,Dataset as torch_Dataset
from fastdatasets.common.iterable_dataset import IterableDatasetBase
from fastdatasets.common.random_dataset import RandomDatasetBase
from .data_writer import DataWriteHelper
from ..utils.maskedlm import make_gpt2_sample
from fastdatasets.leveldb import LEVELDB

__all__ = [
    'DataHelper',
    'make_dataset',
]

class DataPreprocessHelper(object):

    def on_data_ready(self):...

    def on_data_finalize(self):...

    # 下游任务继承
    def on_data_process(self, data: typing.Any, user_data: tuple):
        return make_gpt2_sample(data, user_data)

    def on_task_specific_params(self) -> typing.Dict:
        return {}

    def on_get_labels(self, files: typing.List[str]):
        if not files:
            return None, None
        label_fname = files[0]
        is_json_file = label_fname.endswith('.json')
        D = set()
        with open(label_fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n', '').replace('\n', '')
                if not line: continue
                if is_json_file:
                    jd = json.loads(line)
                    line = jd['label']
                D.add(line)
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label

    # 读取文件
    def on_get_corpus(self, files: typing.List[str], mode: str):
        D = []
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\r\n', '').replace('\n', '')
                    if not line: continue
                    D.append(line)
        return D

class DataTransformHelper(object):
    @staticmethod
    def collate_fn(batch):
        return batch
        # o = {}
        # for i, b in enumerate(batch):
        #     if i == 0:
        #         for k in b:
        #             o[k] = [torch.tensor(b[k])]
        #     else:
        #         for k in b:
        #             o[k].append(torch.tensor(b[k]))
        # for k in o:
        #     o[k] = torch.stack(o[k])
        #
        # max_len = torch.max(o.pop('seqlen'))
        #
        # o['input_ids'] = o['input_ids'][:, :max_len]
        # o['attention_mask'] = o['attention_mask'][:, :max_len]
        # if 'token_type_ids' in o:
        #     o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        # o['labels'] = o['labels'][:, :max_len]
        # return o

    @staticmethod
    def transform(x):
        return x
    @staticmethod
    def batch_transform(batch):
        return batch
        # o = {}
        # for i, b in enumerate(batch):
        #     if i == 0:
        #         for k in b:
        #             o[k] = [torch.tensor(b[k])]
        #     else:
        #         for k in b:
        #             o[k].append(torch.tensor(b[k]))
        # for k in o:
        #     o[k] = torch.stack(o[k])
        #
        # max_len = torch.max(o.pop('seqlen'))
        #
        # o['input_ids'] = o['input_ids'][:, :max_len]
        # o['attention_mask'] = o['attention_mask'][:, :max_len]
        # if 'token_type_ids' in o:
        #     o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        # o['labels'] = o['labels'][:, :max_len]
        # return o


class DataHelper(DataPreprocessHelper,DataTransformHelper):
    def __init__(self,backend: typing.Union[E_file_backend, str],data_process_fn=None,*args,**kwargs):
        DataPreprocessHelper.__init__(self)
        DataTransformHelper.__init__(self)

        self.backend = backend
        self.data_process_fn = self.on_data_process if data_process_fn is None else data_process_fn

    def load_numpy_dataset(self,files: typing.Union[typing.List[str], str],
           options: typing.Union[
               RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions] = None,
           data_key_prefix_list=('input',),
           num_key='total_num',
           cycle_length=1,
           block_length=1):
        return NumpyReaderAdapter.load(files, self.backend , options,
                                       data_key_prefix_list=data_key_prefix_list,
                                       num_key=num_key,
                                       cycle_length=cycle_length,
                                       block_length=block_length)

    """
        cycle_length for IterableDataset
        block_length for IterableDataset
        返回: 
            torch DataLoader
    """
    def load_dataset(self,
                 files: typing.Union[typing.List, str],
                 batch_size: int,
                 num_workers: int = 0,
                 collate_fn:typing.Callable = None,
                 transform_fn: typing.Callable = None,
                 batch_transform_fn: typing.Callable = None,
                 shuffle=False,
                 infinite=False,
                 cycle_length=4, block_length=4):
        if not files:
            return None

        if isinstance(files,str):
            if not os.path.exists(files):
                return None
        else:
            files_ = [f for f in files if f is not None and isinstance(f,str) and os.path.exists(f)]
            if not files_:
                files = [f for f in files if f is not None and isinstance(f,list)]
                if not files:
                    return None
            else:
                files = files_

        dataHelper = self
        dataset = dataHelper.load_numpy_dataset(files, cycle_length=cycle_length, block_length=block_length)

        if isinstance(dataset, typing.Iterator):
            dataset: IterableDatasetBase
            if transform_fn:
                dataset = dataset.apply(transform_fn)
            # if shuffle:
            #     dataset = dataset.shuffle(1024)

            if batch_transform_fn:
                dataset = dataset.batch(batch_size).apply(batch_transform_fn)

            if infinite:
                dataset = dataset.repeat(-1)
            batch_size = batch_size if batch_transform_fn is None else None
            dataset_ = DataLoader(torch_IterableDataset(dataset),
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers,
                                  shuffle=shuffle)

        else:
            dataset: RandomDatasetBase
            if transform_fn:
                dataset = dataset.apply(transform_fn)
            if batch_transform_fn:
                dataset = dataset.batch(batch_size).apply(batch_transform_fn)
            # if shuffle:
            #     dataset = dataset.shuffle(-1)

            batch_size = batch_size if batch_transform_fn is None else None
            dataset_ = DataLoader(torch_Dataset(dataset),
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers,
                                  shuffle=shuffle)
        return dataset_

    def make_dataset(self,outfile: typing.Union[str,list],
                     data,
                     input_fn_args: typing.Tuple,
                     num_process_worker: int = 8,
                     shuffle=True):

        self.on_data_ready()
        fw = DataWriteHelper(self.data_process_fn, input_fn_args,
                             outfile, self.backend, num_process_worker=num_process_worker,
                             shuffle=shuffle)
        fw.save(data)
        self.on_data_finalize()


def make_dataset(data: typing.List,
               input_fn:typing.Callable[[int,typing.Any,tuple],typing.Union[typing.Dict,typing.List,typing.Tuple]],
               input_fn_args:typing.Tuple,
               outfile:str,
               backend: str,
               overwrite = False,
               num_process_worker:int = 8):

    if not os.path.exists(outfile) or overwrite:
        fw = DataWriteHelper(input_fn,input_fn_args,outfile,backend,num_process_worker)
        fw.save(data)