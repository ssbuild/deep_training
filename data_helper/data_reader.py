# @Time    : 2022/11/9 22:55
# @Author  : tk
# @FileName: data_func_args.py
import os

from .data_helper import DataHelper
from .data_module import load_tokenizer, load_configure
from .training_args import ModelArguments, DataArguments, TrainingArguments
from ..utils.func import is_chinese_char

__all__ = [
    'is_chinese_char',
    'get_filename_no_ext',
    'get_filename_replace_dir',
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

    return tokenizer,config,label2id, id2label



