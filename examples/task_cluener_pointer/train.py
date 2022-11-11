# -*- coding: utf-8 -*-
import logging
import os
import sys
from typing import Union, List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../..'))
from pytorch_lightning import Trainer, seed_everything,LightningDataModule
from asmodels.data_helper.data_args_func import make_all_dataset_with_args, load_all_dataset_with_args, load_tokenizer_and_config_with_args
from transformers import AdamW,get_linear_schedule_with_warmup
from asmodels.model.nlp.layers.seq_pointer import EfficientPointerLayer,PointerLayer,loss_fn,f1_metric
from data_loader import NER_DataHelper as DataHelper
from train_args import train_args
from asmodels.model.nlp.models.pointer import TransformerPointer

class MyTransformer(TransformerPointer):
    def __init__(self, *args,**kwargs):
        super(MyTransformer, self).__init__(with_efficient=True,*args,**kwargs)

    def training_step(self, batch, batch_idx):
        labels: torch.Tensor = batch.pop('labels')
        outputs = self(**batch)
        logits = outputs[0]

        logits = self.pointer_layer(logits,batch['attention_mask'])

        loss = loss_fn(labels,logits)
        f1 = f1_metric(labels,logits)
        self.log_dict({
            'train_loss': loss,
            'f1':f1
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels: torch.Tensor = batch.pop('labels')


        real_label = batch.pop(batch)

        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        acc = torch.eq(labels, torch.argmax(outputs[1], dim=1)) / labels.size()[0]
        return {"loss": val_loss, "logits": logits.item(),
                "labels": labels.item(),
                'acc':acc,
                'real_label':real_label
                }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        """Called at the end of the validation epoch with the outputs of all validation steps.

        .. code-block:: python

            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                val_outs.append(out)
            validation_epoch_end(val_outs)

        Args:
            outputs: List of outputs you defined in :meth:`validation_step`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader.

        Return:
            None

        Note:
            If you didn't define a :meth:`validation_step`, this won't be called.

        Examples:
            With a single dataloader:

            .. code-block:: python

                def validation_epoch_end(self, val_step_outputs):
                    for out in val_step_outputs:
                        ...

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each validation step for that dataloader.

            .. code-block:: python

                def validation_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs

                    self.log("final_metric", final_value)
        """

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
