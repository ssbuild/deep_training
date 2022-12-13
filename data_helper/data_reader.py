# @Time    : 2022/11/9 22:55
# @Author  : tk
# @FileName: data_func_args.py
import logging
import os
import typing

from pytorch_lightning import LightningDataModule

from .data_helper import DataHelper
from .training_args import ModelArguments, DataArguments, TrainingArguments
from ..utils.func import is_chinese_char
from .data_module import load_tokenizer,load_configure

__all__ = [
    'is_chinese_char',
    'get_filename_no_ext',
    'get_filename_replace_dir',
    'make_dataset_with_args',
    'load_dataset_with_args',
    'DataHelper',
    'load_tokenizer_and_config_with_args'
]


def get_filename_no_ext(filename):
    filename = os.path.basename(filename)
    pos = filename.rfind('.')
    if pos >= 0:
        filename = filename[:pos]
    return filename


def get_filename_replace_dir(filename,new_path_dir,ext=None):
    return os.path.join(new_path_dir,get_filename_no_ext(filename) + '.' + ext)

def load_tokenizer_and_config_with_args(dataHelper: DataHelper,model_args: ModelArguments,
                                        training_args:TrainingArguments,
                                        data_args: DataArguments,
                                        task_specific_params=None):
    label2id, id2label = dataHelper.on_get_labels(data_args.label_file)
    tokenizer = load_tokenizer(tokenizer_name=model_args.tokenizer_name,
                                model_name_or_path=model_args.model_name_or_path,
                                cache_dir=model_args.cache_dir,
                                do_lower_case=model_args.do_lower_case,
                                use_fast_tokenizer=model_args.use_fast_tokenizer,
                                model_revision=model_args.model_revision,
                                use_auth_token=model_args.use_auth_token,
                               )


    if task_specific_params is None:
        task_specific_params = {}

    task_params = dataHelper.on_task_specific_params()
    if task_params is not None:
        task_specific_params.update(task_params)

    task_specific_params['learning_rate'] = training_args.learning_rate
    task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
        if training_args.learning_rate_for_task is not None else training_args.learning_rate


    kwargs_args = {
        "bos_token_id" : tokenizer.bos_token_id,
        "pad_token_id" : tokenizer.pad_token_id,
        "eos_token_id" : tokenizer.eos_token_id,
        "sep_token_id" : tokenizer.sep_token_id,
        "task_specific_params" : task_specific_params,
    }

    if label2id is not None:
        kwargs_args['label2id'] = label2id
        kwargs_args['id2label'] = id2label
        kwargs_args['num_labels'] = len(label2id) if label2id is not None else None

    config = load_configure(config_name=model_args.config_name,
                            model_name_or_path=model_args.model_name_or_path,
                            cache_dir=model_args.cache_dir,
                            model_revision=model_args.model_revision,
                            use_auth_token=model_args.use_auth_token,
                            **kwargs_args
                            )

    if label2id is not None and hasattr(config,'num_labels'):
        print('*' * 30,'num_labels=', config.num_labels)
        print(label2id)
        print(id2label)

    return  tokenizer,config,label2id, id2label

#返回制作特征数据的中间文件
def get_intermediate_file(data_args:DataArguments, intermediate_name,mode):
    if data_args.data_backend.startswith('memory'):
        #内存数据: list
        intermediate_output = []
        logging.info('make data {} {}...'.format(data_args.output_dir, intermediate_name +  '-' + mode + '.' +  data_args.data_backend))
    else:
        #本地文件数据: 文件名
        intermediate_output = os.path.join(data_args.output_dir, intermediate_name +  '-' + mode + '.' +   data_args.data_backend)
        logging.info('make data {}...'.format(intermediate_output))
    return intermediate_output

def make_dataset_with_args(dataHelper,input_files, fn_args, data_args:DataArguments, intermediate_name, shuffle,mode, num_process_worker=0,overwrite=False):
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
        intermediate_output = get_intermediate_file(data_args,intermediate_name,mode)
        if isinstance(intermediate_output, list) or not os.path.exists(intermediate_output) or overwrite:
            data = dataHelper.on_get_corpus(input_files, mode)
            dataHelper.make_dataset(intermediate_output, data, fn_args, num_process_worker=num_process_worker,shuffle=shuffle)
    else:
        intermediate_output = input_files[0]
    return intermediate_output


class object_for_dataset:
    def __init__(self,train_dataloader,val_dataloader,test_dataloader):
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader


    def train_dataloader(self):
        return self._train_dataloader


    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

'''
   dataHelper: DataHelper
   training_args: args
   allow_train_shuffle: shuffle data for load dataset
'''
def load_dataset_with_args(dataHelper,
                           training_args: TrainingArguments,
                           train_file,
                           eval_file,
                           test_file,
                           allow_train_shuffle=True):

    dataHelper: DataHelper
    dm = LightningDataModule()
    collate_fn,batch_transform,transform = None,None,None
    if dataHelper.collate_fn != DataHelper.collate_fn:
        collate_fn = dataHelper.collate_fn

    if dataHelper.batch_transform != DataHelper.batch_transform:
        batch_transform = dataHelper.batch_transform

    if dataHelper.transform != DataHelper.transform:
        transform = dataHelper.transform

    train_dataloader = dataHelper.load_dataset(train_file,
                                            batch_size=training_args.train_batch_size,
                                            shuffle=allow_train_shuffle,
                                            infinite=True,
                                            transform_fn=transform,
                                            collate_fn=collate_fn,
                                            batch_transform_fn=batch_transform)

    val_dataloader = dataHelper.load_dataset(eval_file,
                                             batch_size=training_args.eval_batch_size,
                                             transform_fn=transform,
                                             collate_fn=collate_fn,
                                             batch_transform_fn=batch_transform)

    test_dataloader = dataHelper.load_dataset(test_file,
                                              batch_size=training_args.test_batch_size,
                                              transform_fn=transform,
                                              collate_fn=collate_fn,
                                              batch_transform_fn=batch_transform)

    obj = object_for_dataset(train_dataloader,val_dataloader,test_dataloader)


    if train_dataloader is not None:
        dm.train_dataloader = obj.train_dataloader
    if val_dataloader is not None:
        dm.val_dataloader = obj.val_dataloader
    if test_dataloader is not None:
        dm.test_dataloader = obj.test_dataloader
    return dm