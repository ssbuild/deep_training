# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02
import json
import logging
import os
import typing

import fastdatasets
from fastdatasets.common.iterable_dataset import IterableDatasetBase
from fastdatasets.common.random_dataset import RandomDatasetBase
from fastdatasets.leveldb import LEVELDB
from fastdatasets.lmdb import LMDB
from fastdatasets.record import RECORD
from fastdatasets import memory as MEMORY
from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset, Dataset as torch_Dataset
from fastdatasets.utils.numpyadapter import NumpyReaderAdapter, E_file_backend

from .training_args import DataArguments
from .data_writer import DataWriteHelper
from ..utils.maskedlm import make_gpt2_sample

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





class DataHelper(DataPreprocessHelper):
    def __init__(self,backend: typing.Union[E_file_backend, str],data_process_fn=None,*args,**kwargs):
        DataPreprocessHelper.__init__(self)

        self.backend = backend
        self.data_process_fn = self.on_data_process if data_process_fn is None else data_process_fn

        self.train_files = []
        self.eval_files = []
        self.test_files = []

    def load_numpy_dataset(self,files: typing.Union[typing.List[str], str],
           options: typing.Union[
               RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions] = None,
           data_key_prefix_list=('input',),
           num_key='total_num',
           cycle_length=1,
           block_length=1,
           with_record_iterable_dataset: bool=False):

        return NumpyReaderAdapter.load(files, self.backend , options,
                                       data_key_prefix_list=data_key_prefix_list,
                                       num_key=num_key,
                                       cycle_length=cycle_length,
                                       block_length=block_length,
                                       with_record_iterable_dataset=with_record_iterable_dataset)

    """
        cycle_length for IterableDataset
        block_length for IterableDataset
        返回: 
            torch DataLoader
    """
    def load_dataset(self,files: typing.Union[typing.List, str],
                     shuffle: bool=False,
                     infinite: bool=False,
                     cycle_length: int=4,
                     block_length: int=10,
                     num_processes: int = 1,
                     process_index: int = 0,
                     with_record_iterable_dataset: bool = False,
                     with_load_memory: bool =False,
                     ):
        assert process_index <= num_processes and num_processes >= 1
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


        dataset = self.load_numpy_dataset(files,
                                          cycle_length=cycle_length,
                                          block_length=block_length,
                                          with_record_iterable_dataset=with_record_iterable_dataset)

        #加载至内存
        if with_load_memory:
            logging.info('load dataset to memory...')
            if isinstance(dataset, typing.Iterator):
                raw_data = [i for i in dataset]
            else:
                raw_data = [dataset[i] for i in range(len(dataset))]

            dataset = MEMORY.load_dataset.SingleRandomDataset(raw_data)

        if isinstance(dataset, typing.Iterator):
            dataset: IterableDatasetBase

            if num_processes > 1:
                dataset = dataset.mutiprocess(num_processes, process_index)

            if shuffle:
                dataset = dataset.shuffle(4096)

            if infinite:
                dataset = dataset.repeat(-1)

            dataset = torch_IterableDataset(dataset)

        else:
            dataset: RandomDatasetBase
            if num_processes > 1:
                dataset = dataset.mutiprocess(num_processes, process_index)

            if shuffle:
                dataset = dataset.shuffle(-1)

            dataset = torch_Dataset(dataset)
        return dataset

    # 返回制作特征数据的中间文件
    def get_intermediate_file(self,data_args: DataArguments, intermediate_name, mode):
        if data_args.data_backend.startswith('memory'):
            # 内存数据: list
            intermediate_output = []
            logging.info('make data {} {}...'.format(data_args.output_dir,
                                                     intermediate_name + '-' + mode + '.' + self.backend))
        else:
            # 本地文件数据: 文件名
            intermediate_output = os.path.join(data_args.output_dir,
                                               intermediate_name + '-' + mode + '.' + self.backend)
            logging.info('make data {}...'.format(intermediate_output))
        return intermediate_output

    def make_dataset_with_args(self, input_files,
                               fn_args: callable,
                               data_args: DataArguments,
                               intermediate_name,
                               shuffle,
                               mode,
                               num_process_worker: int=0,
                               overwrite: bool=False):
        '''
            dataHelper: DataHelper
            save_fn_args: tuple param for DataHelper.on_data_process
            training_args: args
            intermediate_name: str
            allow_train_shuffle: bool， read data is allow shuffle ， but write are in order
            num_process_worker: int , num of process data
        '''
        dataHelper: DataHelper
        if data_args.convert_file:
            intermediate_output = self.get_intermediate_file(data_args, intermediate_name, mode)
            if isinstance(intermediate_output, list) or not os.path.exists(intermediate_output) or overwrite:
                data = self.on_get_corpus(input_files, mode)
                self.make_dataset(intermediate_output,
                                  data,
                                  fn_args,
                                  num_process_worker=num_process_worker,
                                  shuffle=shuffle)
        else:
            intermediate_output = input_files[0]
        return intermediate_output


    def make_dataset(self,outfile: typing.Union[str,list],
                     data,
                     input_fn_args: typing.Tuple,
                     num_process_worker: int = 0,
                     shuffle: bool=True):

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