# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 13:33
import random
import typing

from torch import optim, nn
from torch.optim import AdamW,Adam
from transformers import get_linear_schedule_with_warmup

from ..scheduler import WarmupCosineSchedule
from ...data_helper import TrainingArguments


def configure_optimizers(named_parameter: typing.Union[typing.List,typing.Tuple],
                         training_args: TrainingArguments,
                         estimated_stepping_batches: int):



    if training_args.optimizer.lower() == 'adamw':
        optimizer = AdamW(named_parameter, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    else:
        optimizer = Adam(named_parameter, training_args.learning_rate, eps=training_args.adam_epsilon)

    scheduler = None
    if training_args.scheduler_type.lower() == 'linear'.lower():
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=estimated_stepping_batches
            # num_training_steps=self.trainer.estimated_stepping_batches,
        )
    elif training_args.scheduler_type.lower() == 'CAL'.lower():
        rewarm_epoch_num =  training_args.scheduler["rewarm_epoch_num"]
        eta_min = training_args.scheduler.get('eta_min', 0.)
        last_epoch = training_args.scheduler.get('last_epoch', -1)
        verbose = training_args.scheduler.get('verbose', False)
        T_0 = int(estimated_stepping_batches * rewarm_epoch_num/ training_args.max_epochs)
        T_0 = max(T_0,1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_0,
                                                         eta_min=eta_min,
                                                         last_epoch=last_epoch,
                                                         verbose=verbose)
    elif training_args.scheduler_type.lower() == 'CAWR'.lower():
        T_mult = training_args.scheduler["T_mult"]
        rewarm_epoch_num = training_args.scheduler["rewarm_epoch_num"]
        eta_min = training_args.scheduler.get('eta_min', 0.)
        last_epoch = training_args.scheduler.get('last_epoch', -1)
        verbose = training_args.scheduler.get('verbose', False)
        T_0 = int(estimated_stepping_batches * rewarm_epoch_num / training_args.max_epochs)
        T_0 = max(T_0, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 , T_mult,
                                                                   eta_min=eta_min,
                                                                   last_epoch=last_epoch,
                                                                   verbose=verbose)
    elif training_args.scheduler_type.lower() == 'Step'.lower():
        decay_rate = training_args.scheduler["decay_rate"]
        decay_steps = training_args.scheduler["decay_steps"]
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif training_args.scheduler_type.lower() == 'ReduceLROnPlateau'.lower():
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)
    elif training_args.scheduler_type.lower() == 'WarmupCosine'.lower():
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=training_args.warmup_steps, t_total=estimated_stepping_batches)


    if scheduler:
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    return optimizer



def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串
    """
    random_str =''
    base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length =len(base_str) -1
    for i in range(randomlength):
        random_str +=base_str[random.randint(0, length)]
    return random_str



class InheritBlockMeta(type):
    def __new__(cls, name,bases, attr,*args,**kwargs):
        if kwargs.get('__inherit',False):
            return super(InheritBlockMeta, cls).__new__(cls, name, bases, attr)
        if any(tuple(True for b in bases if issubclass(b,ClassUnBlock))):
            return super(InheritBlockMeta, cls).__new__(cls, name, bases, attr)
        return super(InheritBlockMeta, cls).__new__(cls, name,() if any(tuple(True for b in bases if str(b).endswith('__.ClassBlock\'>'))) else bases, attr)

class ClassUnBlock(metaclass=InheritBlockMeta,__inherit=True):...
class ClassBlock(metaclass=InheritBlockMeta,__inherit=False):...


def block_class(className):
    return type('BC' + generate_random_str(12), (className, ClassBlock,), dict(__MODEL_CLASS__ = className))




class ExceptClassMeta(type):
    def __new__(cls, name,bases,attr,*args,**kwargs):
        excepts = kwargs.pop('except',None)
        return super(ExceptClassMeta, cls).__new__(cls, name,tuple(_ for _ in bases if not str(_).endswith('__.{}\'>'.format(excepts))) if excepts is not None else bases,attr)

class ExceptCLASS(metaclass=ExceptClassMeta):...


def get_value_from_args(key,dtype,*args,**kwargs):
    value = kwargs.get(key, None)
    if value is not None:
        for item in args:
            if isinstance(item,dtype):
                value = item
                break
    assert value is not None, ValueError('no param ',key)
    return value