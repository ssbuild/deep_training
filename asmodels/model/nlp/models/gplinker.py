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

from ..utils import configure_optimizers


class TransformerGplinker(TransformerModel):
    def __init__(self,config, train_args,with_efficient=False, *args,**kwargs):
        super(TransformerGplinker, self).__init__(config, train_args,*args,**kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64,RoPE=False, tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels, 64,RoPE=False,tril_mask=False)

    def configure_optimizers(self):
        attrs = [(self.model, self.config.task_specific_params['learning_rate']),
                 (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
                 (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
                 (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
                 ]
        return configure_optimizers(attrs, self.hparams, self.trainer.estimated_stepping_batches)


    def training_step(self, batch, batch_idx):
        entity_labels: torch.Tensor = batch.pop('entity_labels')
        head_labels: torch.Tensor = batch.pop('head_labels')
        tail_labels: torch.Tensor = batch.pop('tail_labels')
        outputs = self(**batch)
        logits = outputs[0]
        logits1 = self.entities_layer(logits, batch['attention_mask'])
        logits2 = self.heads_layer(logits, batch['attention_mask'])
        logits3 = self.tails_layer(logits, batch['attention_mask'])
        loss = (loss_fn(entity_labels, logits1) +loss_fn(head_labels, logits2) + loss_fn(tail_labels, logits3)) / 3
        f1 = (f1_metric(entity_labels, logits1) +f1_metric(head_labels, logits2) + f1_metric(tail_labels, logits3)) / 3
        self.log_dict({'train_loss': loss, 'f1': f1}, prog_bar=True)
        return loss