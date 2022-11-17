# -*- coding: utf-8 -*-
import logging
import os
import random
import sys

import torch
from torch.nn import CrossEntropyLoss

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))

from torch import nn
from pytorch_lightning import Trainer, seed_everything
from asmodels.data_helper.data_args_func import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import AdamW,get_linear_schedule_with_warmup
from asmodels.model.nlp.models.transformer import TransformerModel
from asmodels.model.nlp.losses.contrast import compute_simcse_loss
from asmodels.model.nlp.layers.mask import unilm_mask
from data_loader import NN_DataHelper as DataHelper
from train_args import train_args

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
    train_args = train_args()
    rng = random.Random(train_args.seed)
    dataHelper = DataHelper(train_args.data_backend)
    tokenizer,config,label2id, id2label = load_tokenizer_and_config_with_args(train_args, dataHelper)

    save_fn_args = (tokenizer, train_args.max_seq_length,
                    rng, train_args.do_whole_word_mask, train_args.max_predictions_per_seq,
                    train_args.masked_lm_prob)

    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = train_args.intermediate_name + '_{}'.format(i)
        logging.info('make data {}...'.format(intermediate_name))
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, train_args,
                                                                      intermediate_name=intermediate_name)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    dm = load_all_dataset_with_args(dataHelper, train_args, train_files, eval_files, test_files,allow_train_shuffle=False)

    dm.setup("fit")
    model = MyTransformer(config=config,train_args=train_args)
    trainer = Trainer(
        # callbacks=[progress_bar],
        max_epochs=train_args.max_epochs,
        max_steps=train_args.max_steps,
        accelerator="gpu",
        devices=1,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=train_args.output_dir,
        gradient_clip_val=train_args.max_grad_norm,
        accumulate_grad_batches = train_args.gradient_accumulation_steps
    )

    if train_args.do_train:
        trainer.fit(model, datamodule=dm)

    if train_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if train_args.do_test:
        trainer.test(model, datamodule=dm)
