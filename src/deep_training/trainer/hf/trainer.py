# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Iterable, List, Optional, Dict, Callable, Tuple

import safetensors
import torch
from accelerate.utils import save_fsdp_model
from torch import nn
from torch.nn import functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, is_torch_tpu_available
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import IS_SAGEMAKER_MP_POST_1_10, TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption, FSDPOption, IntervalStrategy
from transformers.utils import is_peft_available, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, is_sagemaker_mp_enabled, \
    is_accelerate_available
from packaging import version
from ...data_helper.training_args import TrainingArguments, DataArguments, \
    ModelArguments, TrainingArgumentsHF
from ...nlp.models.petl import PetlModel
from ...nlp.models.petl.prompt import PromptModel

from transformers.trainer import logger

if is_peft_available:
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


try:
    from transformers.trainer_pt_utils import remove_dummy_checkpoint
except:
    def remove_dummy_checkpoint(is_main_process, output_dir, filenames):
        if is_main_process:
            for filename in filenames:
                file = os.path.join(output_dir, filename)
                if os.path.isfile(file):
                    os.remove(file)

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
        # _is_peft_model = is_peft_available() and isinstance(model, (PeftModel,PetlModel,PromptModel))

    def get_train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset,DataLoader):
            return self.train_dataset
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = eval_dataset or self.eval_dataset
        if isinstance(eval_dataset, DataLoader):
            return eval_dataset
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if isinstance(test_dataset, DataLoader):
            return test_dataset
        return super().get_test_dataloader(test_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            supported_classes = (PetlModel, PromptModel)
            if is_peft_available():
                supported_classes += (PeftModel,)

            if isinstance(model, supported_classes):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if dataclasses.is_dataclass(outputs):
                loss = outputs.loss
            elif isinstance(outputs,(tuple,list)):
                loss = outputs[0]
            elif isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if isinstance(loss, dict):
                loss = loss["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,PetlModel,PromptModel)
        if is_peft_available():
            supported_classes += (PeftModel,)

        model = self.model
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model.backbone, supported_classes):
            if state_dict is None:
                state_dict = model.state_dict()

            if isinstance(unwrap_model(model), supported_classes):
                unwrap_model(model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.backbone.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                or self.fsdp is not None
                or self.is_fsdp_enabled
        ):
            state_dict = self.model.state_dict() if not self.is_fsdp_enabled else {}
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if self.is_fsdp_enabled:
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)

        elif self.is_deepspeed_enabled:
            # this takes care of everything as long as we aren't under zero3
            if version.parse(accelerate_version) <= version.parse("0.20.3"):
                raise ValueError("Install Accelerate from main branch")
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        parameters = []
        for m, lr in model.get_model_lr():
            decay_parameters = get_parameter_names(m, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            parameters.extend(decay_parameters)
        return parameters

    def get_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        parameters = []
        for m, lr in model.get_model_lr():
            parameter = []
            for n,p in m.named_parameters():
                parameter.append((n,p))
            parameters += parameter
        return parameters

    def get_optimizer_grouped_parameters(self,opt_model):
        decay_parameters = self.get_decay_parameter_names(opt_model)
        parameters = self.get_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in parameters if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in parameters if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(opt_model)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped / 2 ** 20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer