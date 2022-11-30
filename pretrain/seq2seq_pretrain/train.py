# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))

from pytorch_lightning.callbacks import ModelCheckpoint
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from torch.nn import CrossEntropyLoss
from pytorch_lightning import Trainer
from deep_training.data_helper import make_dataset_with_args, load_dataset_with_args, \
    load_tokenizer_and_config_with_args
from deep_training.model.nlp.models.transformer import TransformerForSeq2SeqLM
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_s_len,max_d_len,mode = user_data
        x = data
        o1 = tokenizer.encode_plus(x[0], max_length=max_s_len, truncation=True, add_special_tokens=True, )
        o2 = tokenizer.encode_plus(x[1], max_length=max_d_len, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o1['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o1['attention_mask'], dtype=np.int64)

        slen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_s_len - slen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))

        decoder_input_ids = np.asarray(o2['input_ids'], dtype=np.int64)
        labels = np.asarray(decoder_input_ids[1:],dtype=np.int64)
        decoder_input_ids = np.asarray(decoder_input_ids[:-1],dtype=np.int64)
        decoder_attention_mask = np.asarray([1] * len(decoder_input_ids),dtype=np.int64)

        dlen = np.asarray(len(decoder_input_ids), dtype=np.int64)
        pad_len = max_d_len - dlen
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            decoder_input_ids = np.pad(decoder_input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            decoder_attention_mask = np.pad(decoder_attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))

        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels':labels,
            'slen': slen,
            'dlen': dlen
        }
        return d

    # 读取文件
    @staticmethod
    def read_data_from_file(files: typing.List, mode: str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
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


        slen = torch.max(o.pop('slen'))
        dlen = torch.max(o.pop('dlen'))

        o['input_ids'] = o['input_ids'][:, :slen]
        o['attention_mask'] = o['attention_mask'][:, :slen]

        o['decoder_input_ids'] = o['decoder_input_ids'][:, :dlen]
        o['decoder_attention_mask'] = o['decoder_attention_mask'][:, :dlen]
        o['labels'] = o['labels'][:, :dlen]
        return o


class MyTransformer(TransformerForSeq2SeqLM):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)

    def compute_loss(self,batch) -> tuple:
        labels = None
        if 'labels' in batch:
            labels = batch.pop('labels')
        outputs = self(**batch)
        if labels is not None:
            lm_logits = outputs[0]
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            outputs = (loss,*outputs)
        return outputs


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments,  TrainingArguments,DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer,config,label2id, id2label = load_tokenizer_and_config_with_args(dataHelper,model_args, data_args, training_args)
    save_fn_args = (tokenizer, data_args.max_seq_length,data_args.max_target_length)


    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        logging.info('make data {}...'.format(intermediate_name))
        train_file, eval_file, test_file = make_dataset_with_args(dataHelper, save_fn_args, data_args,
                                                                  intermediate_name=intermediate_name)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    dm = load_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)

    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=5, every_n_train_steps=1000)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1 if data_args.do_eval else None,
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
