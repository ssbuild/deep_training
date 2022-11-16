# @Time    : 2022/11/9 22:55
# @Author  : tk
# @FileName: data_func_args.py
import logging
import os
from pytorch_lightning import LightningDataModule
from .data_helper import DataHelper
from ..utils.utils_func import is_chinese_char
from .data_module import load_tokenizer,load_configure

__all__ = [
    'is_chinese_char',
    'get_filename_no_ext',
    'get_filename_replace_dir',
    'make_all_dataset_with_args',
    'load_all_dataset_with_args',
    'DataHelper',
]


def get_filename_no_ext(filename):
    filename = os.path.basename(filename)
    pos = filename.rfind('.')
    if pos >= 0:
        filename = filename[:pos]
    return filename


def preprocess_args(train_args):
    if train_args.train_file is not None:
        train_args.train_file = train_args.train_file.split(',')

    if train_args.eval_file is not None:
        train_args.eval_file = train_args.eval_file.split(',')

    if train_args.test_file is not None:
        train_args.test_file = train_args.test_file.split(',')

    if train_args.label_file is not None:
        train_args.label_file = train_args.label_file.split(',')
    else:
        train_args.label_file = []
    return train_args


def get_filename_replace_dir(filename,new_path_dir,ext=None):
    return os.path.join(new_path_dir,get_filename_no_ext(filename) + '.' + ext)

def load_tokenizer_and_config_with_args(train_args,dataHelper,task_specific_params=None):
    label2id, id2label = dataHelper.read_labels_from_file(train_args.label_file)
    tokenizer = load_tokenizer(tokenizer_name=train_args.tokenizer_name,
                                model_name_or_path=train_args.model_name_or_path,
                                cache_dir=train_args.cache_dir,
                                do_lower_case=train_args.do_lower_case,
                                use_fast_tokenizer=train_args.use_fast_tokenizer,
                                model_revision=train_args.model_revision,
                                use_auth_token=train_args.use_auth_token,
                               )

    if task_specific_params is None:
        task_specific_params = {}

    task_specific_params['learning_rate'] = train_args.learning_rate
    task_specific_params['learning_rate_for_task'] = train_args.learning_rate_for_task \
        if train_args.learning_rate_for_task is not None else train_args.learning_rate


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

    config = load_configure(config_name=train_args.config_name,
                            model_name_or_path=train_args.model_name_or_path,
                            cache_dir=train_args.cache_dir,
                            model_revision=train_args.model_revision,
                            use_auth_token=train_args.use_auth_token,
                            **kwargs_args
                            )

    return  tokenizer,config,label2id, id2label

def make_all_dataset_with_args(dataHelper,save_fn_args,train_args,intermediate_name,num_process_worker=0):
    dataHelper: DataHelper
    train_file_output, eval_file_output, test_file_output = None, None, None
    if train_args.do_train:
        train_file_output = os.path.join(train_args.output_dir,intermediate_name + '-train.' + train_args.data_backend)
        logging.info('make data {}...'.format(train_file_output))
        train_file_output = dataHelper.make_dataset(train_args.train_file,train_file_output,save_fn_args + ('train',),
                                num_process_worker=num_process_worker,
                                shuffle=True,
                                mode='train')

    if train_args.do_eval:
        eval_file_output = os.path.join(train_args.output_dir,intermediate_name + '-eval.' + train_args.data_backend)
        logging.info('make data {}...'.format(eval_file_output))
        eval_file_output = dataHelper.make_dataset(train_args.eval_file, eval_file_output, save_fn_args+ ('eval',),
                                num_process_worker=num_process_worker,
                                shuffle=False,
                                mode='eval')

    if train_args.do_test:
        test_file_output = os.path.join(train_args.output_dir,intermediate_name + '-test.' + train_args.data_backend)
        logging.info('make data {}...'.format(test_file_output))
        test_file_output = dataHelper.make_dataset(train_args.test_file,test_file_output,save_fn_args+ ('test',),
                                num_process_worker=num_process_worker,
                                shuffle=False,
                                mode='test')
    #特征数据保存至相应的文件或者内存
    return train_file_output, eval_file_output, test_file_output



def load_all_dataset_with_args(dataHelper,train_args,train_file,eval_file,test_file):
    dataHelper: DataHelper
    dm = LightningDataModule()
    train_dataloader = dataHelper.load_dataset(train_file, batch_size=train_args.train_batch_size, shuffle=True,
                                    infinite=True)
    val_dataloader = dataHelper.load_dataset(eval_file, batch_size=train_args.eval_batch_size)
    test_dataloader = dataHelper.load_dataset(test_file, batch_size=train_args.test_batch_size)

    if train_dataloader is not None:
        dm.train_dataloader = lambda: train_dataloader
    if val_dataloader is not None:
        dm.val_dataset = lambda: val_dataloader
    if test_dataloader is not None:
        dm.test_dataset = lambda: test_dataloader
    return dm