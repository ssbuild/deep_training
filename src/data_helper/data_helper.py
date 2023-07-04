# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 11:02
import json
import logging
import os
import typing
import torch
# from fastdatasets import memory as MEMORY
# from fastdatasets.common.iterable_dataset import IterableDatasetBase
# from fastdatasets.common.random_dataset import RandomDatasetBase
# from fastdatasets.torch_dataset import IterableDataset as torch_IterableDataset, Dataset as torch_Dataset
# from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer, PretrainedConfig
from .training_args import ModelArguments, DataArguments, TrainingArguments
from ..utils.func import is_chinese_char
from numpy_io.pytorch_loader.data_helper import DataHelperBase,load_tokenizer, load_configure
from numpy_io.core.writer import DataWriteHelper

__all__ = [
    'DataHelper',
    'is_chinese_char',
    'get_filename_no_ext',
    'get_filename_replace_dir',
    "load_tokenizer",
    "load_configure"
]

def get_filename_no_ext(filename):
    filename = os.path.basename(filename)
    pos = filename.rfind('.')
    if pos >= 0:
        filename = filename[:pos]
    return filename


def get_filename_replace_dir(filename,new_path_dir,ext=None):
    return os.path.join(new_path_dir,get_filename_no_ext(filename) + '.' + ext)


class DataHelper(DataHelperBase):
    tokenizer: typing.Optional[PreTrainedTokenizer] = None
    config: typing.Optional[PretrainedConfig] = None
    model_args: typing.Optional[ModelArguments] = None
    training_args: typing.Optional[TrainingArguments] = None
    data_args: typing.Optional[DataArguments] = None

    def __init__(self,
                 model_args: ModelArguments,
                 training_args: typing.Optional[TrainingArguments] = None,
                 data_args: typing.Optional[DataArguments] = None,
                 **kwargs):

        if data_args:
            super(DataHelper, self).__init__(data_args.data_backend,data_args.convert_file,data_args.output_dir,data_args.intermediate_name)
        else:
            super(DataHelper, self).__init__(None, None, None, None)


        self.label2id = None
        self.id2label = None
        self.max_seq_length_dict = {}
        self._external_kwargs = kwargs
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        if data_args:
            #训练
            label2id, id2label = self.on_get_labels(data_args.label_file)
            self.label2id = label2id
            self.id2label = id2label

            self.max_seq_length_dict['train'] = data_args.train_max_seq_length
            self.max_seq_length_dict['eval'] = data_args.eval_max_seq_length
            self.max_seq_length_dict['val'] = data_args.eval_max_seq_length
            self.max_seq_length_dict['test'] = data_args.test_max_seq_length
            self.max_seq_length_dict['predict'] = data_args.test_max_seq_length
        else:
            #推理
            self.label2id = None
            self.id2label = None

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
                    with_print_config=True,
                    **kwargs):

        model_args = self.model_args
        training_args = self.training_args

        if with_task_params:
            task_specific_params = task_specific_params or {}
            task_params = self.on_task_specific_params()
            if task_params is not None:
                task_specific_params.update(task_params)

            if training_args is not None:
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

        kwargs_args.update(kwargs)

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
                                  with_print_config=True,
                                  tokenizer_kwargs=None,
                                  config_kwargs=None):

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        if config_kwargs is None:
            config_kwargs = {}

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
                                   **tokenizer_kwargs,
                                   )
        self.tokenizer = tokenizer

        if data_args is not None:
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

            if training_args is not None:
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
        kwargs_args.update(config_kwargs)


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
                print('==' * 30, 'num_labels = ', config.num_labels)
                print(self.label2id)
                print(self.id2label)

        if with_labels:
            return tokenizer, config, self.label2id, self.id2label
        return tokenizer, config





