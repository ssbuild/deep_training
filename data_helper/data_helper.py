# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02
import json
import logging
import os
import typing

import torch
from fastdatasets import memory as MEMORY
from fastdatasets.common.iterable_dataset import IterableDatasetBase
from fastdatasets.common.random_dataset import RandomDatasetBase
from fastdatasets.leveldb import LEVELDB
from fastdatasets.lmdb import LMDB
from fastdatasets.record import RECORD
from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset, Dataset as torch_Dataset
from fastdatasets.utils.numpyadapter import NumpyReaderAdapter, E_file_backend
from torch.utils.data import DataLoader, IterableDataset

from .data_module import load_tokenizer, load_configure
from .data_writer import DataWriteHelper
from .training_args import ModelArguments, DataArguments, TrainingArguments
from ..utils.func import is_chinese_char
from ..utils.maskedlm import make_gpt2_sample

__all__ = [
    'DataHelper',
    'make_dataset',
    'is_chinese_char',
    'get_filename_no_ext',
    'get_filename_replace_dir',
]

def get_filename_no_ext(filename):
    filename = os.path.basename(filename)
    pos = filename.rfind('.')
    if pos >= 0:
        filename = filename[:pos]
    return filename


def get_filename_replace_dir(filename,new_path_dir,ext=None):
    return os.path.join(new_path_dir,get_filename_no_ext(filename) + '.' + ext)



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




def check_dataset_file(files):
    if not files:
        return None

    if isinstance(files, str):
        if not os.path.exists(files):
            return None
    else:
        #检测是否是文件list
        files_ = [f for f in files if f is not None and isinstance(f, str) and os.path.exists(f)]
        if not files_:
            #检测是否是内存list
            files = [f for f in files if f is not None and isinstance(f, list)]
            if not files:
                return None
        else:
            files = files_
    return files

class DataHelper(DataPreprocessHelper):
    def __init__(self,model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,**kwargs):
        super(DataHelper, self).__init__()

        self.backend = data_args.data_backend
        self.data_process_fn = self.on_data_process

        self.train_files = []
        self.eval_files = []
        self.test_files = []

        self.tokenizer = None
        self.config = None
        self.label2id = None
        self.id2label = None
        self.model_args =None
        self.training_args = None
        self.data_args = None
        self.max_seq_length_dict = {}


        self._external_kwargs = kwargs

        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        label2id, id2label = self.on_get_labels(data_args.label_file)
        self.label2id = label2id
        self.id2label = id2label

        self.max_seq_length_dict['train'] = data_args.train_max_seq_length
        self.max_seq_length_dict['eval'] = data_args.eval_max_seq_length
        self.max_seq_length_dict['val'] = data_args.eval_max_seq_length
        self.max_seq_length_dict['test'] = data_args.test_max_seq_length
        self.max_seq_length_dict['predict'] = data_args.test_max_seq_length


    @property
    def external_kwargs(self):
        return self._external_kwargs




    def load_tokenizer(self,*args,**kwargs):
        tokenizer = load_tokenizer(*args,**kwargs)
        self.tokenizer = tokenizer
        return tokenizer



    def load_config(self,
                    config_name=None,
                    class_name=None,
                    model_name_or_path=None,
                    task_specific_params=None,
                    with_labels=True,
                    with_task_params=True,
                    return_dict=False,
                    with_print_labels=True,
                    with_print_config=True):

        model_args = self.model_args
        training_args = self.training_args

        if with_task_params:
            task_specific_params = task_specific_params or {}
            task_params = self.on_task_specific_params()
            if task_params is not None:
                task_specific_params.update(task_params)

            task_specific_params['learning_rate'] = training_args.learning_rate
            task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
                if training_args.learning_rate_for_task is not None else training_args.learning_rate


        if hasattr(self.tokenizer,'tokenizer'):
            tokenizer = self.tokenizer
            kwargs_args = {
                "bos_token_id": tokenizer.bos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "sep_token_id": tokenizer.sep_token_id,
                "return_dict": return_dict,
                "task_specific_params": task_specific_params,
            }
        else:
            kwargs_args = {}

        if with_labels and self.label2id is not None:
            kwargs_args['label2id'] = self.label2id
            kwargs_args['id2label'] = self.id2label
            kwargs_args['num_labels'] = len(self.label2id) if self.label2id is not None else None

        config = load_configure(config_name=config_name or model_args.config_name,
                                class_name=class_name,
                                model_name_or_path=model_name_or_path or model_args.model_name_or_path,
                                cache_dir=model_args.cache_dir,
                                model_revision=model_args.model_revision,
                                use_auth_token=model_args.use_auth_token,
                                **kwargs_args
                                )
        self.config = config
        if with_print_config:
            print(config)

        if with_labels and self.label2id is not None and hasattr(config, 'num_labels'):
            if with_print_labels:
                print('*' * 30, 'num_labels = ', config.num_labels)
                print(self.label2id)
                print(self.id2label)
        return config

    def load_tokenizer_and_config(self,
                                  tokenizer_name = None,
                                  config_name = None,
                                  tokenizer_class_name = None,
                                  config_class_name=None,
                                  model_name_or_path = None,
                                  task_specific_params=None,
                                  with_labels= True,
                                  with_task_params=True,
                                  return_dict=False,
                                  with_print_labels=True,
                                  with_print_config=True):

        model_args: ModelArguments = self.model_args
        training_args: TrainingArguments = self.training_args
        data_args: DataArguments = self.data_args



        tokenizer = load_tokenizer(tokenizer_name=tokenizer_name or model_args.tokenizer_name,
                                   class_name=tokenizer_class_name,
                                   model_name_or_path=model_name_or_path or model_args.model_name_or_path,
                                   cache_dir=model_args.cache_dir,
                                   do_lower_case=model_args.do_lower_case,
                                   use_fast_tokenizer=model_args.use_fast_tokenizer,
                                   model_revision=model_args.model_revision,
                                   use_auth_token=model_args.use_auth_token,
                                   )
        self.tokenizer = tokenizer


        self.max_seq_length_dict['train'] = data_args.train_max_seq_length
        self.max_seq_length_dict['eval'] = data_args.eval_max_seq_length
        self.max_seq_length_dict['val'] = data_args.eval_max_seq_length
        self.max_seq_length_dict['test'] = data_args.test_max_seq_length
        self.max_seq_length_dict['predict'] = data_args.test_max_seq_length

        if with_task_params:
            task_specific_params = task_specific_params or {}
            task_params = self.on_task_specific_params()
            if task_params is not None:
                task_specific_params.update(task_params)

            task_specific_params['learning_rate'] = training_args.learning_rate
            task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
                if training_args.learning_rate_for_task is not None else training_args.learning_rate

        kwargs_args = {
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "return_dict": return_dict,
            "task_specific_params": task_specific_params,
        }

        if with_labels and self.label2id is not None:
            kwargs_args['label2id'] = self.label2id
            kwargs_args['id2label'] = self.id2label
            kwargs_args['num_labels'] = len(self.label2id) if self.label2id is not None else None

        config = load_configure(config_name=config_name or model_args.config_name,
                                class_name=config_class_name,
                                model_name_or_path=model_name_or_path or model_args.model_name_or_path,
                                cache_dir=model_args.cache_dir,
                                model_revision=model_args.model_revision,
                                use_auth_token=model_args.use_auth_token,
                                **kwargs_args
                                )
        self.config = config
        if with_print_config:
            print(config)

        if with_labels and self.label2id is not None and hasattr(config, 'num_labels'):
            if with_print_labels:
                print('*' * 30, 'num_labels = ', config.num_labels)
                print(self.label2id)
                print(self.id2label)


        if with_labels:
            return tokenizer, config, self.label2id, self.id2label
        return tokenizer, config


    def load_numpy_dataset(self,files: typing.Union[typing.List[str], str],
           options: typing.Union[
               RECORD.TFRecordOptions, LEVELDB.LeveldbOptions, LMDB.LmdbOptions] = None,
           data_key_prefix_list=('input',),
           num_key='total_num',
           cycle_length=1,
           block_length=1,
           backend=None,
           with_record_iterable_dataset: bool=False,
           with_parse_from_numpy: bool =True,
           limit_start: typing.Optional[int] = None,
           limit_count: typing.Optional[int] = None
                           ):

        dataset = NumpyReaderAdapter.load(files, backend or self.backend , options,
                                       data_key_prefix_list=data_key_prefix_list,
                                       num_key=num_key,
                                       cycle_length=cycle_length,
                                       block_length=block_length,
                                       with_record_iterable_dataset=with_record_iterable_dataset,
                                       with_parse_from_numpy=with_parse_from_numpy)
        if limit_start is not None and limit_start >0:
            dataset = dataset.skip(limit_start)
        if limit_count is not None and limit_count > 0:
            dataset = dataset.limit(limit_count)
        return dataset

    """
        cycle_length for IterableDataset
        block_length for IterableDataset
        return: 
            torch DataLoader or fastdatasets numpy dataset
    """
    def load_dataset(self,files: typing.Union[typing.List, str],
                     shuffle: bool=False,
                     infinite: bool=False,
                     cycle_length: int=4,
                     block_length: int=10,
                     num_processes: int = 1,
                     process_index: int = 0,
                     backend=None,
                     with_record_iterable_dataset: bool = False,
                     with_load_memory: bool = False,
                     with_torchdataset: bool = True,
                     transform_fn : typing.Callable = None,
                     check_dataset_file_fn=None,
                     limit_start: typing.Optional[int] = None,
                     limit_count: typing.Optional[int] = None
                     ) -> typing.Optional[typing.Union[torch.utils.data.Dataset,torch.utils.data.IterableDataset]]:
        assert process_index <= num_processes and num_processes >= 1
        check_dataset_file_fn = check_dataset_file_fn or check_dataset_file
        files = check_dataset_file_fn(files)
        if files is None:
            return None

        dataset = self.load_numpy_dataset(files,
                                          cycle_length=cycle_length,
                                          block_length=block_length,
                                          with_record_iterable_dataset=with_record_iterable_dataset,
                                          with_parse_from_numpy=not with_load_memory,
                                          backend=backend,
                                          limit_start=limit_start,
                                          limit_count=limit_count)
        #加载至内存
        if with_load_memory:
            logging.info('load dataset to memory...')
            if isinstance(dataset, typing.Iterator):
                raw_data = [i for i in dataset]
            else:
                raw_data = [dataset[i] for i in range(len(dataset))]

            dataset = MEMORY.load_dataset.SingleRandomDataset(raw_data)
            #解析numpy数据
            if self.backend != 'memory_raw':
                dataset = dataset.parse_from_numpy_writer()


        if isinstance(dataset, typing.Iterator):
            dataset: IterableDatasetBase
            if num_processes > 1:
                dataset = dataset.mutiprocess(num_processes, process_index)

            if shuffle:
                dataset = dataset.shuffle(4096)

            if infinite:
                dataset = dataset.repeat(-1)

            if transform_fn is not None:
                dataset = dataset.map(transform_fn)

            dataset_ = torch_IterableDataset(dataset) if with_torchdataset else dataset
        else:
            dataset: RandomDatasetBase
            if num_processes > 1:
                dataset = dataset.mutiprocess(num_processes, process_index)

            if shuffle:
                dataset = dataset.shuffle(-1)

            if transform_fn is not None:
                dataset = dataset.map(transform_fn)

            dataset_ = torch_Dataset(dataset) if with_torchdataset else dataset
        return dataset_


    def load_random_sampler(self,files: typing.Union[typing.List, str],
                     batch_size,
                     collate_fn=None,
                     pin_memory=False,
                     shuffle: bool=False,
                     infinite: bool=False,
                     cycle_length: int=4,
                     block_length: int=10,
                     num_processes: int = 1,
                     process_index: int = 0,
                     backend=None,
                     with_record_iterable_dataset: bool = False,
                     with_load_memory: bool = False,
                     with_torchdataset: bool = True,
                     transform_fn : typing.Callable = None,
                     check_dataset_file_fn=None,
                    limit_start: typing.Optional[int] = None,
                    limit_count: typing.Optional[int] = None
                    ) -> typing.Optional[typing.Union[DataLoader,torch.utils.data.Dataset,torch.utils.data.IterableDataset,IterableDatasetBase,RandomDatasetBase]]:

        dataset = self.load_dataset(
            files,shuffle=shuffle,infinite=infinite,cycle_length=cycle_length,
            block_length=block_length,num_processes=num_processes,process_index=process_index,
            backend=backend,with_record_iterable_dataset=with_record_iterable_dataset,
            with_load_memory=with_load_memory,with_torchdataset=with_torchdataset,
            transform_fn=transform_fn,check_dataset_file_fn=check_dataset_file_fn,
            limit_start=limit_start,
            limit_count=limit_count
        )
        if dataset is None:
            return None
        return DataLoader(dataset,batch_size=batch_size,shuffle=False if isinstance(dataset, IterableDataset) else shuffle,collate_fn=collate_fn,pin_memory=pin_memory)

    def load_sequential_sampler(self,files: typing.Union[typing.List, str],
                     batch_size,
                     collate_fn=None,
                     pin_memory=False,
                     shuffle: bool=False,
                     infinite: bool=False,
                     cycle_length: int=4,
                     block_length: int=10,
                     num_processes: int = 1,
                     process_index: int = 0,
                     backend=None,
                     with_record_iterable_dataset: bool = False,
                     with_load_memory: bool = False,
                     with_torchdataset: bool = True,
                     transform_fn : typing.Callable = None,
                     check_dataset_file_fn=None,
                    limit_start: typing.Optional[int] = None,
                    limit_count: typing.Optional[int] = None
                                ) -> typing.Optional[typing.Union[DataLoader,torch.utils.data.Dataset,torch.utils.data.IterableDataset,IterableDatasetBase,RandomDatasetBase]]:

        dataset = self.load_dataset(
            files,shuffle=shuffle,infinite=infinite,cycle_length=cycle_length,
            block_length=block_length,num_processes=num_processes,process_index=process_index,
            backend=backend,with_record_iterable_dataset=with_record_iterable_dataset,
            with_load_memory=with_load_memory,with_torchdataset=with_torchdataset,
            transform_fn=transform_fn,check_dataset_file_fn=check_dataset_file_fn,
            limit_start=limit_start,
            limit_count=limit_count
        )
        if dataset is None:
            return None
        return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,pin_memory=pin_memory)

    # 返回制作特征数据的中间文件
    def get_intermediate_file(self, intermediate_name, mode):
        data_args: DataArguments = self.data_args
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
                               shuffle,
                               mode,
                               num_process_worker: int=0,
                               overwrite: bool=False,
                               dupe_factor=1):
        '''
            save_fn_args: tuple param for DataHelper.on_data_process
            training_args: args
            intermediate_name: str
            allow_train_shuffle: bool， read data is allow shuffle ， but write are in order
            num_process_worker: int , num of process data
        '''

        data_args: DataArguments = self.data_args
        for i in range(dupe_factor):
            intermediate_name = data_args.intermediate_name + '_{}'.format(i)
            if data_args.convert_file:
                intermediate_output = self.get_intermediate_file(intermediate_name, mode)
                if isinstance(intermediate_output, list) or not os.path.exists(intermediate_output) or overwrite:
                    data = self.on_get_corpus(input_files, mode)
                    self.make_dataset(intermediate_output,
                                      data,
                                      mode,
                                      num_process_worker=num_process_worker,
                                      shuffle=shuffle)
            else:
                intermediate_output = input_files[0]

            if mode == 'train':
                self.train_files.append(intermediate_output)
            elif mode == 'eval' or mode == 'val':
                self.eval_files.append(intermediate_output)
            elif mode == 'test' or mode == 'predict':
                self.test_files.append(intermediate_output)
            else:
                raise ValueError('{} invalid ',mode)



    def make_dataset(self,outfile: typing.Union[str,list],
                     data,
                     input_fn_args: typing.Any,
                     num_process_worker: int = 0,
                     shuffle: bool=True):

        self.on_data_ready()
        fw = DataWriteHelper(self.data_process_fn,
                             input_fn_args,
                             outfile,
                             self.backend,
                             num_process_worker=num_process_worker,
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