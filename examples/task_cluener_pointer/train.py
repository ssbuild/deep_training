# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
import logging
from typing import Union, List
import torch
import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from asmodels.model.nlp.layers.seq_pointer import f1_metric

from pytorch_lightning import Trainer, seed_everything,LightningDataModule
from asmodels.data_helper.data_args_func import make_all_dataset_with_args, load_all_dataset_with_args, load_tokenizer_and_config_with_args

from data_loader import NER_DataHelper as DataHelper
from train_args import train_args
from asmodels.model.nlp.models.pointer import TransformerPointer
from asmodels.model.nlp.metrics.pointer import metric_for_pointer

class MyTransformer(TransformerPointer):
    def __init__(self, *args,**kwargs):
        super(MyTransformer, self).__init__(with_efficient=True,*args,**kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels: torch.Tensor = batch.pop('labels')
        real_label = batch.pop("real_label")
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        f1 = f1_metric(labels,logits)
        return {"loss": val_loss, "logits": logits.item(),"labels": real_label,'f1':f1}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        id2label = self.config.id2label
        threshold = 1e-7
        preds = []
        labels = []
        for o in outputs:
            logits = o['logits']
            label = o['labels']
            for tag in logits:
                one_result = []
                for (l, s, e) in zip(*np.where(tag > threshold)):
                    one_result.append((l, s, e))
                preds.append(one_result)
            labels.append(label)
        m = metric_for_pointer(labels,preds,id2label)
        print(m)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        out = self(x)
        return out



if __name__== '__main__':
    train_args = train_args()
    seed_everything(train_args.seed)
    if not os.path.exists(train_args.output_dir):
        os.mkdir(train_args.output_dir)

    dataHelper = DataHelper(train_args.data_backend)
    tokenizer,config,label2id, id2label = load_tokenizer_and_config_with_args(train_args, dataHelper)
    save_fn_args = (tokenizer, train_args.max_seq_length,label2id)


    N = 1
    train_files, eval_files, test_files = [], [], []
    for i in range(N):
        intermediate_name = train_args.intermediate_name + '_{}'.format(i)
        train_file, eval_file, test_file = make_all_dataset_with_args(dataHelper, save_fn_args, train_args,
                                                                      intermediate_name=intermediate_name,num_process_worker=0)
        train_files.append(train_file)
        eval_files.append(eval_file)
        test_files.append(test_file)


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
