# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 13:33
import typing

from transformers import AdamW, get_linear_schedule_with_warmup


def configure_optimizers(model_attrs: typing.Union[typing.List,typing.Tuple],
                         hparams: typing.Dict,
                         estimated_stepping_batches: int):
    no_decay = ["bias", "LayerNorm.weight"]
    # attrs = [(model, self.config.task_specific_params['learning_rate']),
    #          (self.classifier, self.config.task_specific_params['learning_rate']),
    #          (self.crf, self.config.task_specific_params['learning_rate_for_task']), ]
    opt = []
    for a, lr in model_attrs:
        opt += [
            {
                "params": [p for n, p in a.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": hparams.weight_decay, "lr": lr,
            },
            {
                "params": [p for n, p in a.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, "lr": lr,
            },
        ]
    optimizer = AdamW(opt, lr=hparams.learning_rate, eps=hparams.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hparams.warmup_steps,
        num_training_steps=estimated_stepping_batches
        # num_training_steps=self.trainer.estimated_stepping_batches,
    )
    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]