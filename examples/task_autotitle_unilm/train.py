# -*- coding: utf-8 -*-
import json
import logging
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import BertTokenizer
from deep_training.model.nlp.models.transformer import TransformerModelUnilm
from transformers import HfArgumentParser
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.utils.func import seq_pading

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,mode = user_data
        x = data
        assert isinstance(x,tuple)
        o = tokenizer.encode_plus(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True)
        input_array = [o['input_ids'],o['token_type_ids']]
        seqlen = np.asarray(input_array[0],dtype=np.int64)
        input_ids ,token_type_ids = seq_pading(input_array,max_seq_length=max_seq_length,pad_val=tokenizer.pad_token_id)
        d = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'seqlen': seqlen
        }
        return d


    # 读取文件
    @staticmethod
    def read_data_from_file(files:typing.List,mode:str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i,line in enumerate(lines):
                    jd = json.loads(line)
                    D.append((jd['content'], jd['title']))
                    if i > 1000:
                        break
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
        o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        return o


class MyTransformer(TransformerModelUnilm):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length)


    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        logging.info('make data {}...'.format(intermediate_name))
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, data_args,
                                                                      intermediate_name=intermediate_name)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)

    dm.setup("fit")
    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    trainer = Trainer(
        # callbacks=[progress_bar],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches = training_args.gradient_accumulation_steps
    )

    if data_args.do_train:
        trainer.fit(model, datamodule=dm)

    if data_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if data_args.do_test:
        trainer.test(model, datamodule=dm)
