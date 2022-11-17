# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import argparse
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss

from deep_training.data_helper.training_args import TrainingArguments, ModelArguments
from ..layers.mask import lm_mask,unilm_mask

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForVisualQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForSemanticSegmentation,
    AutoModelForObjectDetection,
    AutoModelForAudioClassification,
    AutoModelForMaskedImageModeling,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup, AutoModelForPreTraining, AutoModel,
)

from ..utils import configure_optimizers


class TransformerBase(LightningModule):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__()


        if hasattr(config,'task_specific_params') or config.task_specific_params is None:
            config.task_specific_params = {}
        task_specific_params = config.task_specific_params
        task_specific_params['learning_rate'] = training_args.learning_rate
        task_specific_params['learning_rate_for_task'] = training_args.learning_rate_for_task \
            if training_args.learning_rate_for_task is not None else training_args.learning_rate

        print(training_args)
        print(model_args)
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.model_args = model_args
        self.training_args = training_args

    def get_model_lr(self):
        return [(self.model, self.config.task_specific_params['learning_rate']), ]  if hasattr(self,'model') else []

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        return configure_optimizers(self.get_model_lr(), self.hparams,self.trainer.estimated_stepping_batches)


    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss',loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        labels = batch["labels"]
        return {"losses": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        out = self(x)
        return out
        # losses = self.losses(out, y)
        #
        # # log 6 example images
        # # or generated text... or whatever
        # sample_imgs = x[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('example_images', grid, 0)
        #
        # # calculate acc
        # labels_hat = torch.argmax(out, dim=1)
        # test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        #
        # # log the outputs!
        # self.log_dict({'test_loss': losses, 'test_acc': test_acc})
    #
    # def validation_epoch_end(self, outputs):
    #     # logits = torch.cat([x["logits"] for x in outputs]).detach().cpu().numpy()
    #     # labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    #     losses = torch.stack([x["losses"] for x in outputs]).mean()
    #     self.log("val_loss", losses, prog_bar=True)
    #     # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)


class TransformerModel(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config, model_args,training_args, *args, **kwargs)
        config = self.config

        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModel.from_config(config)
        self.model = model

    def get_model_lr(self):
        return [(self.model, self.config.task_specific_params['learning_rate'])]

class TransformerModelUnilm(TransformerModel):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config, model_args,training_args, *args, **kwargs)
        config = self.config

        self.loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_model_lr(self):
        return super(TransformerModelUnilm, self).get_model_lr() + \
            [(self.lm_head,self.config.task_specific_params['learning_rate_for_task']),]

    def training_step(self, batch, batch_idx):
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        if 'labels' in batch:
            labels = batch.pop('labels')
        else:
            labels = batch['input_ids']
        outputs = self(**batch)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.log('train_loss',loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch['attention_mask'] = unilm_mask(batch['token_type_ids'])
        if 'labels' in batch:
            labels = batch.pop('labels')
        else:
            labels = batch['input_ids']
        outputs = self(**batch)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        val_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"losses": val_loss, "logits": lm_logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x['attention_mask'] = unilm_mask(x['token_type_ids'])
        outputs = self(**batch)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

class TransformerForPreTraining(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config, model_args,training_args, *args, **kwargs)
        config = self.config

        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForPreTraining.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForPreTraining.from_config(config)
        self.model = model

class TransformerForCausalLM(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        

        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_config(config)
        self.model = model




class TransformerForMaskLM(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForMaskedLM.from_config(config)
        self.model = model



class TransformerForSeq2SeqLM(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_config(config)
        self.model = model


class TransformerForSequenceClassification(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForSequenceClassification.from_config(config)
        self.model = model


class TransformerForQuestionAnswering(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForQuestionAnswering.from_config(config)
        self.model = model


class TransformerForVisualQuestionAnswering(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForVisualQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForVisualQuestionAnswering.from_config(config)
        self.model = model






class TransformerForTokenClassification(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForTokenClassification.from_config(config)
        self.model = model


class TransformerForMultipleChoice(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForMultipleChoice.from_config(config)
        self.model = model

class TransformerForNextSentencePrediction(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForNextSentencePrediction.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForNextSentencePrediction.from_config(config)
        self.model = model


class TransformerForImageClassification(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForImageClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForImageClassification.from_config(config)
        self.model = model


class TransformerForImageSegmentation(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForImageSegmentation.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForImageSegmentation.from_config(config)
        self.model = model



class TransformerForSemanticSegmentation(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForSemanticSegmentation.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForSemanticSegmentation.from_config(config)
        self.model = model

class TransformerForObjectDetection(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForObjectDetection.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForObjectDetection.from_config(config)
        self.model = model



class TransformerForAudioClassification(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForAudioClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForAudioClassification.from_config(config)
        self.model = model


class TransformerForMaskedImageModeling(TransformerBase):
    def __init__(self, config,model_args:ModelArguments, training_args:TrainingArguments, *args: Any, **kwargs: Any):
        super().__init__(config,model_args,training_args, *args, **kwargs)
        config = self.config
        
        if model_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": True if model_args.use_auth_token else None,
            }
            model = AutoModelForMaskedImageModeling.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForMaskedImageModeling.from_config(config)
        self.model = model
