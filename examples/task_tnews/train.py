# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))

import torch
import logging
from pytorch_lightning import Trainer
from deep_training.data_helper.data_args_func import make_all_dataset_with_args, load_all_dataset_with_args, \
    load_tokenizer_and_config_with_args
from transformers import AdamW, get_linear_schedule_with_warmup, HfArgumentParser
from deep_training.model.nlp.models.transformer import TransformerForSequenceClassification
from data_loader import NN_DataHelper as DataHelper
from deep_training.data_helper.training_args import ModelArguments, TrainingArguments, DataArguments

class MyTransformer(TransformerForSequenceClassification):
    def __init__(self,*args,**kwargs):
        super(MyTransformer, self).__init__(*args,**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss,logits = outputs[0:2]
        labels: torch.Tensor = batch['labels']
        acc = torch.sum(torch.eq(labels.view(-1),torch.argmax(logits,dim=1,keepdim=False))) / labels.view(-1).size()[0]
        self.log_dict({
            'train_loss': loss,
            'acc':acc
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        labels = batch['labels']
        acc = torch.eq(labels, torch.argmax(outputs[1], dim=1)) / labels.size()[0]
        return {"losses": val_loss, "logits": logits, "labels": labels,'acc':acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        out = self(x)
        return out



if __name__== '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    dataHelper = DataHelper(data_args.data_backend)
    tokenizer, config, label2id, id2label = load_tokenizer_and_config_with_args(dataHelper, model_args, data_args,
                                                                                training_args)
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
    dm = load_all_dataset_with_args(dataHelper, data_args, train_files, eval_files, test_files)

    dm.setup("fit")
    model = MyTransformer(config=config,model_args=model_args,training_args=training_args)
    trainer = Trainer(
        # callbacks=[progress_bar],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,  # limiting got iPython runs
        enable_progress_bar=True,
        default_root_dir=training_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches = training_args.gradient_accumulation_steps
    )

    if training_args.do_train:
        trainer.fit(model, datamodule=dm)

    if training_args.do_eval:
        trainer.validate(model, datamodule=dm)

    if training_args.do_test:
        trainer.test(model, datamodule=dm)
