# -*- coding: utf-8 -*-
import json
import os
import sys
import typing
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from pytorch_lightning.callbacks import ModelCheckpoint
from deep_training.data_helper import DataHelper
import torch
import numpy as np
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args

from deep_training.model.nlp.models.pointer import TransformerForPointer
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments

train_info_args = {
    'devices': '1',
    'data_backend': 'memory_raw',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    'train_file': '/data/nlp/nlp_train_data/clue/cluener/train.json',
    'eval_file': '/data/nlp/nlp_train_data/clue/cluener/dev.json',
    'test_file': '/data/nlp/nlp_train_data/clue/cluener/test.json',
    'learning_rate': 5e-5,
    'max_epochs': 15,
    'train_batch_size': 64,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 160
}


def convert_feature(data, user_data):
    tokenizer: BertTokenizer
    tokenizer, max_seq_length, label2id, mode = user_data
    sentence, label_dict = data

    input_ids = tokenizer.convert_tokens_to_ids(list(sentence))
    if len(input_ids) > max_seq_length - 2:
        input_ids = input_ids[:max_seq_length - 2]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    seqlen = np.asarray(len(input_ids), dtype=np.int32)
    labels = np.zeros(shape=(len(label2id), max_seq_length, max_seq_length), dtype=np.int32)
    real_label = []
    for label_str, o in label_dict.items():
        pts = list(o.values())[0]
        labelid = label2id[label_str]
        for pt in pts:
            if pt[1] < max_seq_length:
                labels[labelid, pt[0], pt[1]] = 1
            real_label.append((labelid, pt[0], pt[1]))

    pad_len = max_seq_length - len(input_ids)
    if pad_len > 0:
        pad_val = tokenizer.pad_token_id
        input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
    d = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'seqlen': seqlen,
    }
    # if mode == 'eval':
    #     d['real_label'] = np.asarray(bytes(json.dumps(real_label, ensure_ascii=False), encoding='utf-8'))
    return d


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        return convert_feature(data, user_data)

    # 读取标签
    @staticmethod
    def read_labels_from_file(label_fname: str):
        labels = [
            'address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene'
        ]
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        return label2id, id2label

    # 读取文件
    @staticmethod
    def read_data_from_file(files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['text'], jd.get('label', None)))
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

        o['labels'] = o['labels'][:, :, :max_len, :max_len]
        return o


class MyTransformer(TransformerForPointer):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length, label2id)

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, data_args,
                                                                      intermediate_name=intermediate_name,
                                                                      num_process_worker=0)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)

    model = MyTransformer(with_efficient=True,config=config, model_args=model_args, training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", save_last=True, every_n_epochs=1)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps
    )

    if data_args.do_train:
        trainer.fit(model, datamodule=dm)

    if data_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if data_args.do_test:
        trainer.test(model, datamodule=dm)
