# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
import typing

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'..'))
import numpy as np
from deep_training.data_helper import DataHelper
from deep_training.data_helper import ModelArguments, TrainingArguments, PrefixModelArguments, \
    DataArguments
import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import HfArgumentParser, BertTokenizer
from deep_training.model.nlp.models.prefixtuning import PrefixTransformerForSequenceClassification

train_info_args = {
    'devices':  '1',
    'data_backend': 'leveldb',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'do_eval': True,
    'train_file': '/data/nlp/nlp_train_data/clue/tnews/train.json',
    'eval_file': '/data/nlp/nlp_train_data/clue/tnews/dev.json',
    'test_file': '/data/nlp/nlp_train_data/clue/tnews/test.json',
    'label_file': '/data/nlp/nlp_train_data/clue/tnews/labels.json',
    'learning_rate': 5e-5,
    'max_epochs': 3,
    'train_batch_size': 10,
    'test_batch_size': 2,
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 512,
    'pre_seq_len': 16
}

class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self,data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length,label2id,pre_seq_len,mode = user_data
        sentence,label_str = data

        max_seq_length -= pre_seq_len

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

    #读取标签
    @staticmethod
    def read_labels_from_file(files: str):
        if not files:
            return None, None

        D = set()
        for label_fname in files:
            is_json_file = label_fname.endswith('.json')
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


class MyTransformer(PrefixTransformerForSequenceClassification):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(prompt_type=0,*args,**kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)

    def compute_loss(self,batch) -> tuple:
        labels: torch.Tensor = batch.pop('labels',None)
        outputs = self.get_transformer_outputs(batch)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            acc = torch.sum(torch.eq(labels.view(-1),torch.argmax(logits,dim=1,keepdim=False))) / labels.view(-1).size()[0]
            loss_dict = {
                'train_loss': loss,
                'acc':acc
            }
            outputs = (loss_dict,logits,labels)
        else:
            outputs = (logits,)
        return outputs

    def validation_epoch_end(self, outputs: typing.Union[EPOCH_OUTPUT, typing.List[EPOCH_OUTPUT]]) -> None:
        preds_all, labels_all = [], []
        for output in outputs:
            preds, labels = output['outputs']
            preds = np.argmax(preds,-1)
            for p, l in zip(preds, labels):
                preds_all.append(p)
                labels_all.append(int(l))

        preds_all = np.asarray(preds_all,dtype=np.int32)
        labels_all = np.asarray(labels_all, dtype=np.int32)
        f1 = f1_score(labels_all, preds_all, average='micro')
        report = classification_report(labels_all, preds_all, digits=4,
                                       labels=list(self.config.label2id.values()),target_names=list(self.config.label2id.keys()))

        print(f1, report)
        self.log('val_f1', f1)


if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PrefixModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args, prompt_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args, prompt_args = parser.parse_args_into_dataclasses()

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    save_fn_args = (tokenizer, data_args.max_seq_length,label2id,training_args.pre_seq_len)

    print(label2id,id2label)
    print('*' * 30,config.num_labels)

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

    
    model = MyTransformer(prompt_args=prompt_args,config=config,model_args=model_args,training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", save_last=True, every_n_epochs=1)
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
