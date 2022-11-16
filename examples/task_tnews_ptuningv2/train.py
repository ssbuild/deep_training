# -*- coding: utf-8 -*-
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))

import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from pytorch_lightning import Trainer, seed_everything,LightningDataModule
from asmodels.data_helper.data_args_func import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import AdamW,get_linear_schedule_with_warmup
from asmodels.model.nlp.models.ptuingv2 import PrefixTransformerForSequenceClassification
from data_loader import NN_DataHelper as DataHelper
from train_args import train_args

class MyTransformer(PrefixTransformerForSequenceClassification):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)

    def training_step(self, batch, batch_idx):
        labels: torch.Tensor = batch.pop('labels')
        outputs = self.get_transformer_outputs(batch)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        acc = torch.sum(torch.eq(labels.view(-1),torch.argmax(logits,dim=1,keepdim=False))) / labels.view(-1).size()[0]
        self.log_dict({
            'train_loss': loss,
            'acc':acc
        }, prog_bar=True)
        return loss


if __name__== '__main__':
    train_args = train_args()
    seed_everything(train_args.seed)
    if not os.path.exists(train_args.output_dir):
        os.mkdir(train_args.output_dir)

    dataHelper = DataHelper(train_args.data_backend)
    tokenizer,config,label2id, id2label = load_tokenizer_and_config_with_args(train_args, dataHelper)
    save_fn_args = (tokenizer, train_args.max_seq_length,label2id)


    print(label2id)
    print(id2label)
    print('*' * 30,config.num_labels)

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

    print(train_files, eval_files, test_files)
    dm = load_all_dataset_with_args(dataHelper, train_args, train_files, eval_files, test_files)

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
