# -*- coding: utf-8 -*-
import json
import logging
import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'..'))
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from torch import nn
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from deep_training.model.nlp.models.transformer import TransformerModelUnilm
from deep_training.model.nlp.losses.contrast import compute_simcse_loss
from deep_training.model.nlp.layers.mask import unilm_mask
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments


train_info_args = {
    'device' '1' 
    'data_backend': 'leveldb',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'train_file': '/data/nlp/nlp_train_data/thucnews/train.json',
    'max_steps': 100000,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir' : './output',
    'max_seq_length' : 512,
    'max_target_length' : 50
}

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,mode = user_data
        x = data
        assert isinstance(x,tuple)

        outputs = []
        for _ in range(2):
            o = tokenizer(text=x[0], text_pair=x[1], max_length=max_seq_length, truncation=True,
                          add_special_tokens=True)

            input_ids = np.asarray(o['input_ids'], dtype=np.int64)
            token_type_ids = np.asarray(o['token_type_ids'], dtype=np.int64)

            seqlen = np.asarray(len(input_ids), dtype=np.int64)
            pad_len = max_seq_length - len(input_ids)
            if pad_len > 0:
                pad_val = tokenizer.pad_token_id
                input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
                token_type_ids = np.pad(token_type_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            d = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'labels': input_ids,
                'seqlen': seqlen
            }
            outputs.append(d)
        return outputs


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
        return D[0:1000] if mode == 'train' else D[:100]


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
        o['labels'] = o['labels'][:, :max_len]
        return o


class MyTransformer(TransformerModelUnilm):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def compute_loss(self,batch):
        labels = batch.pop('labels',None)
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        outputs = self(**batch)
        lm_logits = self.lm_head(outputs[0])
        simcse_logits = self.sim_head(outputs[1])

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss1 = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss2 = compute_simcse_loss(simcse_logits)
            loss = loss1 + loss2
            loss_dict = {
                'loss': loss,
                'unilm_loss': loss1,
                'simcse_loss': loss2,
            }
            outputs = (loss_dict,lm_logits,simcse_logits)
            self.log_dict(loss_dict, prog_bar=True)
        else:
            outputs = (lm_logits,simcse_logits)
        return outputs



if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)

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

    dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files,allow_train_shuffle=False)

    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="loss", save_last=True, every_n_epochs=1)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
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
