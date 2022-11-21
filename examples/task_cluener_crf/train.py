# -*- coding: utf-8 -*-
import json
import os
import sys
import typing

from pytorch_lightning.callbacks import ModelCheckpoint, Checkpoint, LambdaCallback
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
from deep_training.data_helper import DataHelper
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from deep_training.model.nlp.models.crf_model import TransformerForCRF
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):

        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,mode = user_data
        sentence,label_dict = data

        input_ids = tokenizer.convert_tokens_to_ids(list(sentence))
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:max_seq_length - 2]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        seqlen = np.asarray(len(input_ids), dtype=np.int64)

        labels = np.zeros(shape=(seqlen,),dtype=np.int64)

        for label_str, o in label_dict.items():
            pts = list(o.values())[0]
            for pt in pts:
                if pt[1] > seqlen - 2:
                    continue
                pt[0] += 1
                pt[1] += 1
                span_len = pt[1] - pt[0] + 1
                if span_len == 1:
                    labels[pt[0]] = label2id['S_' + label_str]
                elif span_len == 2:
                    labels[pt[0]] = label2id['B_' + label_str]
                    labels[pt[1]] = label2id['E_' + label_str]
                    for i in range(span_len - 2):
                        labels[pt[0] + 1 + i] = label2id['I_' + label_str]


        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen,
        }
        return d

    #读取标签
    @staticmethod
    def read_labels_from_file(label_fname: str):
        labels = [
            'address','book','company','game','government','movie','name','organization','position','scene'
        ]
        labels = ['O'] + [t + '_' + l  for t in ['B','I','E','S'] for l in labels]
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
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
                    D.append((jd['text'], jd.get('label',None)))
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

        o['labels'] = o['labels'][:,:max_len]
        return o

class MyTransformer(TransformerForCRF):
    def __init__(self, *args,**kwargs):
        super(MyTransformer, self).__init__(with_efficient=True,*args,**kwargs)


class MyModelCheckpoint(Callback):
    def __init__(self,model,eval_dm,*args,**kwargs):
        self.model = model
        self.eval_dm = eval_dm
        super(Callback, self).__init__(*args,**kwargs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") :
        pass
        # result = trainer.validate(self.model,self.eval_dm)
        # print(result)
        # print('*' * 30)


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)

    save_fn_args = (tokenizer, data_args.max_seq_length,label2id)


    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, data_args,
                                                                      intermediate_name=intermediate_name,num_process_worker=0)
        train_files.append(train_file[:1000])
        eval_files.append(eval_file[:100])
        test_files.append(test_file)

    train_dm,eval_dm,test_dm = load_all_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)
    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    #checkpoint_callback = MyModelCheckpoint(eval_dm,save_top_k=0,every_n_epochs=1)
    # checkpoint_callback = MyModelCheckpoint(model,eval_dm)

    #[batch, seq] , [batch, seq]
    #[batch * num_tags, seq] , [batch* num_tags, seq]
    def on_train_epoch_end(trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pl_module.eval()
        result = trainer.validate(model, eval_dm)
        print(result)

    checkpoint_callback = LambdaCallback(on_train_epoch_end=on_train_epoch_end)
    trainer = Trainer(
        log_every_n_steps = 10,
        callbacks=[checkpoint_callback],
        # check_val_every_n_epoch=1 if data_args.do_eval else None,
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches = training_args.gradient_accumulation_steps
    )

    #【batch,num_label,seq,seq】
    #[batch,num_label *  seq * 2 ]
    if data_args.do_train and train_dm:
        trainer.fit(model, datamodule=train_dm)

    if data_args.do_eval and eval_dm:
        trainer.validate(model, datamodule=eval_dm)

    if data_args.do_test and test_dm:
        trainer.test(model, datamodule=test_dm)
