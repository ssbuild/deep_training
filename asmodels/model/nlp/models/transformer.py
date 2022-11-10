# @Time    : 2022/11/6 19:30
# @Author  : tk
# @FileName: model.py

import argparse
from typing import Any
from pytorch_lightning import LightningModule

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
    get_linear_schedule_with_warmup,
)

class TransformerBase(LightningModule):
    def __init__(self, config,train_args, *args: Any, **kwargs: Any):
        super().__init__()
        save_args = train_args._get_kwargs()
        save_args = {
            item[0]: item[1]
            for item in save_args
        }
        print(save_args)
        self.save_hyperparameters(save_args,ignore='config')
        self.config = config

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


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
        return {"loss": val_loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        out = self(x)
        return out
        # loss = self.loss(out, y)
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
        # self.log_dict({'test_loss': loss, 'test_acc': test_acc})
    #
    # def validation_epoch_end(self, outputs):
    #     # logits = torch.cat([x["logits"] for x in outputs]).detach().cpu().numpy()
    #     # labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     self.log("val_loss", loss, prog_bar=True)
    #     # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

class TransformerForCausalLM(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args, *args, **kwargs)
        config = self.config
        

        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForCausalLM.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_config(config)
        self.model = model

class TransformerForMaskLM(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForMaskedLM.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForMaskedLM.from_config(config)
        self.model = model



class TransformerForSeq2SeqLM(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }

            model = AutoModelForSeq2SeqLM.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_config(config)
        self.model = model


class TransformerForSequenceClassification(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForSequenceClassification.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForSequenceClassification.from_config(config)
        self.model = model


class TransformerForQuestionAnswering(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForQuestionAnswering.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForQuestionAnswering.from_config(config)
        self.model = model


class TransformerForVisualQuestionAnswering(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForVisualQuestionAnswering.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForVisualQuestionAnswering.from_config(config)
        self.model = model






class TransformerForTokenClassification(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForTokenClassification.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForTokenClassification.from_config(config)
        self.model = model


class TransformerForMultipleChoice(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForMultipleChoice.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForMultipleChoice.from_config(config)
        self.model = model

class TransformerForNextSentencePrediction(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForNextSentencePrediction.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForNextSentencePrediction.from_config(config)
        self.model = model


class TransformerForImageClassification(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForImageClassification.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForImageClassification.from_config(config)
        self.model = model


class TransformerForImageSegmentation(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForImageSegmentation.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForImageSegmentation.from_config(config)
        self.model = model



class TransformerForSemanticSegmentation(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForSemanticSegmentation.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs
            )
        else:
            model = AutoModelForSemanticSegmentation.from_config(config)
        self.model = model

class TransformerForObjectDetection(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForObjectDetection.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForObjectDetection.from_config(config)
        self.model = model



class TransformerForAudioClassification(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForAudioClassification.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForAudioClassification.from_config(config)
        self.model = model


class TransformerForMaskedImageModeling(TransformerBase):
    def __init__(self,config,train_args: argparse.Namespace, *args: Any, **kwargs: Any):
        super().__init__(config,train_args,*args,**kwargs)
        config = self.config
        
        if train_args.model_name_or_path:
            model_kwargs = {
                "cache_dir": train_args.cache_dir,
                "revision": train_args.model_revision,
                "use_auth_token": True if train_args.use_auth_token else None,
            }
            model = AutoModelForMaskedImageModeling.from_pretrained(
                train_args.model_name_or_path,
                from_tf=bool(".ckpt" in train_args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            model = AutoModelForMaskedImageModeling.from_config(config)
        self.model = model

