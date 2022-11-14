# @Time    : 2022/11/11 20:15
# @Author  : tk
# @FileName: gp_linker.py
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, loss_fn, f1_metric

__all__ = [
    'TransformerGplinker'
]

class TransformerGplinker(TransformerModel):
    def __init__(self,with_efficient=False, *args,**kwargs):
        super(TransformerGplinker, self).__init__(*args,**kwargs)

        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64,RoPE=False, tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size,
                                              self.config.num_labels,
                                              64,
                                              RoPE=False,
                                              tril_mask=False)



    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        attrs = [model,self.entities_layer,self.heads_layer,self.tails_layer]
        opt = []
        for a in attrs:
            opt += [
                    {
                        "params": [p for n, p in a.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.hparams.weight_decay,"lr": self.hparams.learning_rate,
                    },
                    {
                        "params": [p for n, p in a.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,"lr": self.hparams.learning_rate,
                    },
                ]


        optimizer = AdamW(opt, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]



    def training_step(self, batch, batch_idx):
        entity_labels: torch.Tensor = batch.pop('entity_labels')
        head_labels: torch.Tensor = batch.pop('head_labels')
        tail_labels: torch.Tensor = batch.pop('tail_labels')
        outputs = self(**batch)
        logits = outputs[0]



        logits1 = self.entities_layer(logits, batch['attention_mask'])
        logits2 = self.heads_layer(logits, batch['attention_mask'])
        logits3 = self.tails_layer(logits, batch['attention_mask'])

        print(entity_labels.shape)
        print(head_labels.shape)
        print(tail_labels.shape)

        print(logits1.shape)
        print(logits2.shape)
        print(logits3.shape)

        loss = loss_fn(entity_labels, logits1) +loss_fn(head_labels, logits2) + loss_fn(tail_labels, logits3)
        f1 = (f1_metric(entity_labels, logits1) +f1_metric(head_labels, logits2) + f1_metric(tail_labels, logits3)) / 3
        self.log_dict({'train_loss': loss, 'f1': f1}, prog_bar=True)
        return loss