# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import random
import torch
import logging
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from transformers import AdamW,get_linear_schedule_with_warmup
from asmodels.model.nlp.models.transformer import TransformerForMaskLM
from data_loader import MLM_DataHelper as DataHelper
from asmodels.dataHelper.data_args_func import load_tokenizer_and_config_with_args, make_all_dataset_with_args, load_all_dataset_with_args
from train_args import build_args

class MyTransformer(TransformerForMaskLM):
    def __init__(self,tokenizer,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)
        self.loss_fct = CrossEntropyLoss(reduction='none',ignore_index=tokenizer.pad_token_id)  # -100 index = padding token

    def _compute_loss(self,y_trues,y_preds,weight):
        y_preds = torch.transpose(y_preds, 1, 2)
        loss = self.loss_fct(y_preds,y_trues)
        loss = loss * weight
        loss = torch.sum(loss, dtype=torch.float) / (
                    torch.sum(weight, dtype=torch.float) + 1e-8)
        return loss

    def training_step(self, batch, batch_idx):
        weight = batch.pop('weight')
        labels = batch.pop('labels')
        outputs = self(**batch)
        logits = outputs[0]
        loss = self._compute_loss(labels,logits,weight)
        self.log('batch_idx',batch_idx,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        weight = batch.pop('weight')
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        labels = batch["labels"]
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        weight = batch.pop('weight')
        if 'labels' in batch:
            batch.pop('labels')
        x, y = batch
        out = self(x)
        return out

if __name__== '__main__':
    train_args = build_args()
    rng = random.Random(train_args.random_seed)
    seed_everything(train_args.seed)
    if not os.path.exists(train_args.output_dir):
        os.mkdir(train_args.output_dir)

    dataHelper = DataHelper(train_args.data_backend)
    tokenizer, config = load_tokenizer_and_config_with_args(train_args, dataHelper)
    save_fn_args = (tokenizer, train_args.max_seq_length,
                    rng, train_args.do_whole_word_mask, train_args.max_predictions_per_seq,
                    train_args.masked_lm_prob)

    N = train_args.dupe_factor
    train_files,eval_files,test_files = [],[],[]
    for i in range(N):
        intermediate_name = train_args.intermediate_name + '_{}'.format(i)
        logging.info('make data {}...'.format(intermediate_name))
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, train_args,intermediate_name=intermediate_name)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)

    print(train_files, eval_files, test_files)
    dm = load_all_dataset_with_args(dataHelper, train_args, train_files, eval_files, test_files)
    dm.setup("fit")
    model = MyTransformer(tokenizer,config=config,train_args=train_args)
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