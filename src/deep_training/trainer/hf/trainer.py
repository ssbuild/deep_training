# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
from dataclasses import dataclass, field
from typing import Union, Iterable, List, Optional, Dict, Callable, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback
from transformers.trainer_utils import ShardedDDPOption, FSDPOption, IntervalStrategy
from ...data_helper.training_args import TrainingArguments, DataArguments, \
    ModelArguments, TrainingArgumentsHF


# def convert2hf_args(train_args: TrainingArgumentsDT,
#                     model_args: ModelArgumentsDT,
#                     data_args: DataArgumentsDT,
#                     eval_delay=0,
#
#
#                     ):
#     args = dict(
#         output_dir=data_args.output_dir,
#         overwrite_output_dir=data_args.overwrite_cache,
#         do_train=data_args.do_train,
#         do_eval=data_args.do_eval,
#         do_predict=data_args.do_test,
#         per_device_train_batch_size=train_args.train_batch_size,
#         per_device_eval_batch_size=train_args.eval_batch_size,
#         gradient_accumulation_steps=train_args.gradient_accumulation_steps,
#         eval_accumulation_steps=train_args.gradient_accumulation_steps,
#         eval_delay=eval_delay,
#         learning_rate=train_args.learning_rate,
#         weight_decay=train_args.weight_decay,
#         adam_beta1 = train_args.optimizer_betas[0],
#         adam_beta2 = train_args.optimizer_betas[1],
#         adam_epsilon=train_args.adam_epsilon,
#         max_grad_norm=train_args.max_grad_norm,
#         num_train_epochs=train_args.max_epochs,
#         max_steps=train_args.max_steps,
#         lr_scheduler_type=train_args.scheduler_type,
#         warmup_steps=train_args.warmup_steps,
#         log_level=log_level,
#         log_level_replica=log_level_replica,
#         log_on_each_node=log_on_each_node,
#         logging_dir=logging_dir,
#         logging_strategy=logging_strategy,
#         logging_first_step=logging_first_step,
#         logging_steps=logging_steps,
#         logging_nan_inf_filter=logging_nan_inf_filter,
#
#     )


class TrainerHF(Trainer):
    def __init__(self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArgumentsHF = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs):
        super().__init__(model=model,
                         args = args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         )