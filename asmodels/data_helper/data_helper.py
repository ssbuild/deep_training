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
from fastdatasets.utils.NumpyAdapter import NumpyReaderAdapter,E_file_backend
from torch.utils.data import DataLoader
from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset,Dataset as torch_Dataset
from fastdatasets.common.iterable_dataset import IterableDatasetBase
from fastdatasets.common.random_dataset import RandomDatasetBase
from .data_writer import DataWriteHelper
from ..utils.nlpfn import make_gpt2_sample

__all__ = [
    'DataHelper',
    'make_dataset',
]



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


class DataHelper:
    def __init__(self,backend: typing.Union[E_file_backend, str]):
        self.backend = backend

    def load_numpy_dataset(self,files: typing.Union[typing.List[str], str],
           options: typing.Union[
               RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions] = None,
           data_key_prefix_list=('input',),
           num_key='total_num',
           cycle_length=1,
           block_length=1):
        return NumpyReaderAdapter.load(files, self.backend , options,
                                data_key_prefix_list=data_key_prefix_list, num_key=num_key,
                                cycle_length=cycle_length, block_length=block_length)

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
                 transform_fn: typing.Callable = None,
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



        collate_fn = dataHelper.collect_fn
        if isinstance(dataset, typing.Iterator):
            dataset: IterableDatasetBase
            if transform_fn:
                dataset = dataset.apply(transform_fn)
            if shuffle:
                dataset = dataset.shuffle(1024)
            if infinite:
                dataset = dataset.repeat(-1)

            dataset_ = DataLoader(torch_IterableDataset(dataset),
                                  batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

        else:
            dataset: RandomDatasetBase
            if transform_fn:
                dataset = dataset.apply(transform_fn)
            if shuffle:
                dataset = dataset.shuffle(-1)
            dataset_ = DataLoader(torch_Dataset(dataset),
                                  batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
        return dataset_

    #自定义函数
    def make_dataset_custom(self,data: typing.List,
                 input_fn: typing.Callable[
                     [int, typing.Any, tuple], typing.Union[typing.Dict, typing.List, typing.Tuple]],
                 input_fn_args: typing.Tuple,
                 outfile: str,
                 overwrite=False,
                 num_process_worker: int = 8,
                shuffle=True):
        if not os.path.exists(outfile) or overwrite:
            fw = DataWriteHelper(input_fn, input_fn_args, outfile, self.backend,
                                 num_process_worker=num_process_worker,shuffle=shuffle)
            outfile = fw.save(data)
        return outfile

    #回调 on_data_process
    def make_dataset(self,input_files: typing.Union[typing.List[str],str],
                     outfile: str,
                     input_fn_args: typing.Tuple,
                     overwrite=False,
                     num_process_worker: int = 8,
                     shuffle=True,
                     mode=None):
        if not os.path.exists(outfile) or overwrite:
            data = self.read_data_from_file(input_files,mode)
            fw = DataWriteHelper(self.on_data_process, input_fn_args,
                                 outfile, self.backend, num_process_worker=num_process_worker,shuffle=shuffle)
            outfile = fw.save(data)
        return outfile

    #下游任务继承
    def on_data_process(self,data: typing.Any, user_data: tuple):
        return make_gpt2_sample(data,user_data)

    @staticmethod
    def read_labels_from_file(files: typing.List[str]):
        if not files:
            return None,None
        label_fname = files[0]
        is_json_file = label_fname.endswith('.json')
        D = set()
        with open(label_fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r\n', '').replace('\n','')
                if not line: continue
                if is_json_file:
                    jd = json.loads(line)
                    line = jd['label']
                D.add(line)
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label
    # 读取文件
    @staticmethod
    def read_data_from_file(files:typing.List[str],mode:str):
        D = []
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('\r\n', '').replace('\n')
                    if not line: continue
                    D.append(line)
        return D

    @staticmethod
    def collect_fn(batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlen = o.pop('seqlen')
        max_len = torch.max(seqlen)

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        return o







