# @Time    : 2022/11/11 20:15
# @Author  : tk
# @FileName: gp_linker.py
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from .transformer import TransformerModel
from ..layers.seq_pointer import EfficientPointerLayer, PointerLayer, loss_fn, f1_metric

__all__ = [
    'TransformerForGplinker'
]



class TransformerForGplinker(TransformerModel):
    def __init__(self,with_efficient=False, *args,**kwargs):
        super(TransformerForGplinker, self).__init__(*args, **kwargs)
        PointerLayerObject = EfficientPointerLayer if with_efficient else PointerLayer
        self.entities_layer = PointerLayerObject(self.config.hidden_size, 2, 64)
        self.heads_layer = PointerLayerObject(self.config.hidden_size, self.config.num_labels, 64,RoPE=False, tril_mask=False)
        self.tails_layer = PointerLayerObject(self.config.hidden_size,self.config.num_labels, 64,RoPE=False,tril_mask=False)

    def get_model_lr(self):
        return super(TransformerForGplinker, self).get_model_lr() + [
            (self.entities_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.heads_layer, self.config.task_specific_params['learning_rate_for_task']),
            (self.tails_layer, self.config.task_specific_params['learning_rate_for_task']),
        ]

    def comput_loss(self,batch):

        outputs = self(**batch)
        logits = outputs[0]
        logits1 = self.entities_layer(logits, batch['attention_mask'])
        logits2 = self.heads_layer(logits, batch['attention_mask'])
        logits3 = self.tails_layer(logits, batch['attention_mask'])
        if 'entity_labels' in batch:
            entity_labels: torch.Tensor = batch.pop('entity_labels')
            head_labels: torch.Tensor = batch.pop('head_labels')
            tail_labels: torch.Tensor = batch.pop('tail_labels')

            loss = (loss_fn(entity_labels, logits1) + loss_fn(head_labels, logits2) + loss_fn(tail_labels, logits3)) / 3
            f1 = (f1_metric(entity_labels, logits1) + f1_metric(head_labels, logits2) + f1_metric(tail_labels, logits3)) / 3
            loss_dict = {'train_loss': loss, 'f1': f1}
            outputs = (loss_dict,logits1,logits2,logits3)
        else:
            outputs = (logits1,logits2,logits3)
        return outputs