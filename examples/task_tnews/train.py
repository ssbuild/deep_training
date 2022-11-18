# -*- coding: utf-8 -*-
import json
import os
import sys
import typing

from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import numpy as np
from deep_training.data_helper import DataHelper
import torch
import logging
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from deep_training.model.nlp.models.transformer import TransformerForSequenceClassification
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self,data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,mode = user_data
        sentence,label_str = data

        o = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=True, )
        input_ids = np.asarray(o['input_ids'], dtype=np.int64)
        attention_mask = np.asarray(o['attention_mask'], dtype=np.int64)

        labels = np.asarray(label2id[label_str] if label_str is not None else 0,dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen
        }
        return d

    # 读取标签
    @staticmethod
    def read_labels_from_file(files: str):
        if files is None:
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
    @staticmethod
    def read_data_from_file(files: typing.List,mode:str):
        D = []
        for filename in files:
            with open(filename, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    D.append((jd['sentence'], jd.get('label',None)))
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
        return o



class MyTransformer(TransformerForSequenceClassification):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        if 'labels' in batch:
            loss,logits = outputs[0:2]
            labels: torch.Tensor = batch['labels']
            acc = torch.sum(torch.eq(labels.view(-1),torch.argmax(logits,dim=1,keepdim=False))) / labels.view(-1).size()[0]
            loss_dict = {
                'train_loss': loss,
                'acc':acc
            }
            outputs = (loss_dict,logits)
        else:
            outputs = (outputs[0],)
        return outputs


        return out



if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length,label2id)

    print(label2id, id2label)
    print('*' * 30, config.num_labels)

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

    print(train_files, eval_files, test_files)
    dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)

    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True, every_n_epochs=1)
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
