# -*- coding: utf-8 -*-
import logging
import os
import random
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import typing

import torch
from torch.nn import CrossEntropyLoss
from deep_training.data_helper import DataHelper
from torch import nn
from pytorch_lightning import Trainer
from deep_training.data_helper import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from deep_training.model.nlp.models.transformer import TransformerModel
from deep_training.model.nlp.losses.contrast import compute_simcse_loss
from transformers import HfArgumentParser, BertTokenizer
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments,MlmDataArguments
from deep_training.utils.wwm import make_mlm_wwm_sample


class NN_DataHelper(DataHelper):
    # 切分词
    def on_data_process(self, data: typing.Any, user_data: tuple):
        tokenizer: BertTokenizer
        tokenizer,max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob,mode = user_data
        # assert isinstance(data,tuple)
        documents = data
        document_text_string = ''.join(documents)
        document_texts = []
        pos = 0
        while pos < len(document_text_string):
            text = document_text_string[pos:pos + max_seq_length - 2]
            pos += len(text)
            document_texts.append(text)
        # 返回多个文档
        document_nodes = []
        for text in document_texts:
            for _ in range(2):
                node = make_mlm_wwm_sample(text, tokenizer,max_seq_length, rng, do_whole_word_mask, max_predictions_per_seq, masked_lm_prob)
                document_nodes.append(node)
        return document_nodes

    @staticmethod
    def read_data_from_file(input_files: typing.List, mode: str):
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

        seqlen = o.pop('seqlen')
        max_len = torch.max(seqlen)

        o['input_ids'] = o['input_ids'][:, :max_len]
        o['attention_mask'] = o['attention_mask'][:, :max_len]
        if 'token_type_ids' in o:
            o['token_type_ids'] = o['token_type_ids'][:, :max_len]
        o['labels'] = o['labels'][:, :max_len]
        o['weight'] = o['weight'][:, :max_len]
        return o


class MyTransformer(TransformerModel):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sim_head = nn.Linear(config.hidden_size, 512, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.config.pad_token_id)

    def get_model_lr(self):
        return super(MyTransformer, self).get_model_lr() + [
            (self.mlm_head, self.config.task_specific_params['learning_rate_for_task']),
            (self.sim_head, self.config.task_specific_params['learning_rate_for_task'])
        ]

    def _compute_loss(self, y_trues, y_preds, weight):
        y_preds = torch.transpose(y_preds, 1, 2)
        loss = self.loss_fct(y_preds, y_trues)
        loss = loss * weight
        loss = torch.sum(loss, dtype=torch.float) / (torch.sum(weight, dtype=torch.float) + 1e-8)
        return loss

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        weight = batch.pop('weight')
        outputs = self(**batch)
        mlm_logits = self.mlm_head(outputs[0])
        simcse_logits = self.sim_head(outputs[1])
        loss1 = self._compute_loss(labels,mlm_logits,weight)
        loss2 = compute_simcse_loss(simcse_logits)
        loss = loss1 + loss2
        self.log_dict({
            'mlm_loss': loss1,
            'simcse_loss': loss2,
            'train_loss': loss
        },prog_bar=True)
        return loss

if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments,MlmDataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args,mlm_data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args,mlm_data_args = parser.parse_args_into_dataclasses()

    dataHelper = NN_DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, training_args,data_args)
    rng = random.Random(training_args.seed)
    save_fn_args = (tokenizer, data_args.max_seq_length,
                    rng, mlm_data_args.do_whole_word_mask, mlm_data_args.max_predictions_per_seq,mlm_data_args.masked_lm_prob)

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
