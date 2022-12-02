# -*- coding: utf-8 -*-
import json
import os
import sys

from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import typing
from deep_training.data_helper import DataHelper
from deep_training.utils.wwm import make_mlm_wwm_sample
import random
import torch
import logging
from torch.nn import CrossEntropyLoss
from pytorch_lightning import Trainer
from transformers import BertTokenizerFast,HfArgumentParser
from deep_training.model.nlp.models.transformer import TransformerForMaskLM, TransformerMeta
from deep_training.data_helper import load_tokenizer_and_config_with_args, make_dataset_with_args, \
    load_dataset_with_args
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments,MlmDataArguments


train_info_args = {
    'devices':  '1',
    'data_backend': 'leveldb',
    'model_type': 'bert',
    'model_name_or_path': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'tokenizer_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese',
    'config_name': '/data/nlp/pre_models/torch/bert/bert-base-chinese/config.json',
    'do_train': True,
    'train_file': '/data/nlp/nlp_train_data/thucnews/train.json',
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
    'train_max_seq_length': 512,
    'eval_max_seq_length': 512,
    'test_max_seq_length': 512,
    'do_lower_case': False,
    'do_whole_word_mask': True,
    'max_predictions_per_seq': 20,
    'dupe_factor': 5,
    'masked_lm_prob': 0.15
}




class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: typing.Any):
        tokenizer: BertTokenizerFast
        tokenizer,max_seq_length,do_lower_case, label2id,\
        rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob,mode = user_data

        documents = data
        document_text_string = ''.join(documents)
        document_texts = []
        pos = 0
        while pos < len(document_text_string):
            text = document_text_string[pos:pos + max_seq_length - 2]
            pos += len(text)
            document_texts.append(text)
        #返回多个文档
        document_nodes = []
        for text in document_texts:
            node = make_mlm_wwm_sample(text, tokenizer,max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob)
            document_nodes.append(node)
        return document_nodes


    # 读取文件
    @staticmethod
    def read_data_from_file(input_files: typing.List,mode:str):
        D = []
        line_no = 0
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    jd = json.loads(line)
                    if not jd:
                        continue
                    text = jd['content']
                    docs = text.split('\n\n')
                    D.append([doc for doc in docs if doc])
                    line_no += 1

                    if line_no > 1000:
                        break

                    if line_no % 10000 == 0:
                        print('read_line', line_no)
                        print(D[-1])
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

        max_len = torch.max(o.pop('seqlen'))

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        o['weight'] = o['weight'][:, :max_len]
        return o

class MyTransformer(TransformerForMaskLM,metaclass=TransformerMeta):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.loss_fct = CrossEntropyLoss(reduction='none',ignore_index=self.config.pad_token_id)

    def compute_loss_mlm(self,y_trues,y_preds,weight):
        y_preds = torch.transpose(y_preds, 1, 2)
        loss = self.loss_fct(y_preds,y_trues)
        loss = loss * weight
        loss = torch.sum(loss, dtype=torch.float) / (torch.sum(weight, dtype=torch.float) + 1e-8)
        return loss.mean()

    def compute_loss(self,batch) -> tuple:
        labels,weight = None,None
        if 'labels' in batch:
            weight = batch.pop('weight')
            labels = batch.pop('labels')
        outputs = self(**batch)
        logits = outputs[0]
        if labels is not  None:
            loss = self.compute_loss_mlm(labels,logits,weight)
            outputs = (loss,logits,labels)
        else:
            outputs = (logits, )
        return outputs


if __name__== '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,MlmDataArguments))
    model_args, training_args, data_args, mlm_data_args = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,
                                                                                data_args)

    rng = random.Random(training_args.seed)
    token_fn_args_dict = {
        'train': (tokenizer, data_args.train_max_seq_length, model_args.do_lower_case, label2id,
                  rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                  mlm_data_args.masked_lm_prob,
                  'train'),
        'eval': (tokenizer, data_args.eval_max_seq_length, model_args.do_lower_case, label2id,
                 rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                 mlm_data_args.masked_lm_prob,
                 'eval'),
        'test': (tokenizer, data_args.test_max_seq_length, model_args.do_lower_case, label2id,
                 rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,
                 mlm_data_args.masked_lm_prob,
                 'test')
    }


    N = mlm_data_args.dupe_factor
    train_files,eval_files,test_files = [],[],[]
    for i in range(N):
        intermediate_name = data_args.intermediate_name + '_{}'.format(i)
        if data_args.do_train:
            train_files.append(
                make_dataset_with_args(dataHelper, data_args.train_file, token_fn_args_dict['train'], data_args,
                                       intermediate_name=intermediate_name, shuffle=True, mode='train'))
        if data_args.do_eval:
            eval_files.append(
                make_dataset_with_args(dataHelper, data_args.eval_file, token_fn_args_dict['eval'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='eval'))
        if data_args.do_test:
            test_files.append(
                make_dataset_with_args(dataHelper, data_args.test_file, token_fn_args_dict['test'], data_args,
                                       intermediate_name=intermediate_name, shuffle=False, mode='test'))

    print(train_files, eval_files, test_files)
    dm = load_dataset_with_args(dataHelper, training_args, train_files, eval_files, test_files)
    
    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    checkpoint_callback = ModelCheckpoint(monitor="loss", save_top_k=10, every_n_train_steps=1000)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  
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