# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 15:23

import argparse

from deep_training.data_helper.data_args_func import preprocess_args


def train_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')


    parser.add_argument('--data_backend', default='record', type=str, help='record,leveldb,lmdb,memory,memory_raw')

    parser.add_argument('--model_name_or_path', default=None, type=str, help='模型参数配置信息')
    parser.add_argument('--model_type', default=None, type=str, help='模型参数配置信息')
    parser.add_argument('--config_overrides', default=None, type=str, help= "Override some existing default config_gpt2 settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")


    parser.add_argument('--tokenizer_name', default=None, type=str, help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--do_lower_case', default=True, type=bool,
                        help='Whether to lower case the input text. Should be True for uncased deep_training and False for cased deep_training.')

    parser.add_argument('--config_name', default=None, type=str, help='Pretrained config_gpt2 name or path if not the same as model_name"')
    parser.add_argument('--cache_dir', default=None, type=str,
                        help='Where do you want to store the pretrained deep_training downloaded from huggingface.co')
    parser.add_argument('--use_fast_tokenizer', default=True, type=bool,
                        help='Where do you want to store the pretrained deep_training downloaded from huggingface.co')
    parser.add_argument('--model_revision', default="main", type=str,
                        help='The specific model version to use (can be a branch name, tag name or commit id).')
    parser.add_argument('--use_auth_token', default=False, type=bool,
                        help='The specific model version to use (can be a branch name, tag name or commit id).')

    parser.add_argument('--learning_rate', default=5e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--learning_rate_for_task', default=None, type=float, help='下游模型训练时的学习率')

    parser.add_argument('--max_epochs', default=-1, type=int, help='模型训练的轮数')
    parser.add_argument('--max_steps', default=-1, type=int, help='max_steps')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='')
    parser.add_argument('--warmup_steps', default=0, type=int, help='')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')


    parser.add_argument('--max_seq_length', type=int, default=512, help='输入模型的最大长度')
    parser.add_argument('--max_target_length', type=int, default=64, help='生成标题的最大长度，要比max_len小')

    parser.add_argument('--do_train', action="store_true", help='是否训练')
    parser.add_argument('--do_eval', action="store_true", help='是否eval')
    parser.add_argument('--do_test', action="store_true", help='是否test')
    parser.add_argument('--train_file', default='./data_dir/train_data.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--eval_file', default=None, type=str,help='新闻标题生成的训练数据')
    parser.add_argument('--test_file', default=None, type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--label_file', default=None, type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--intermediate_name', default='feature', type=str,
                        help='dataset文件名前缀')

    parser.add_argument('--train_batch_size', default=16, type=int, help='训练时每个batch的大小')
    parser.add_argument('--eval_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')



    parser.add_argument('--output_dir', default='output_dir', type=str, help='模型输出路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return preprocess_args(parser.parse_args())
