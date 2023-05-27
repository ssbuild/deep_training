# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 13:33
import math
import random
import typing
import torch
from torch import optim
from ..optimizer.optimizer import get_optimizer_cls_and_kwargs
from ..scheduler import WarmupCosineSchedule
from ..scheduler.scheduler import get_scheduler, SchedulerType
from ...data_helper import TrainingArguments

try:
    from transformers import AdamW as AdamWHF
except:
    AdamWHF = None




def configure_optimizers(optimizer_grouped_parameters: typing.Union[typing.List,typing.Tuple],
                         args: TrainingArguments,
                         estimated_stepping_batches: int,
                         model_attrs = None
                         ):
    num_training_steps = estimated_stepping_batches
    optimizer_name = args.optimizer.lower()
    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(optimizer_name, args)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        for (opt_model,lr) in model_attrs:
            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, torch.nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    # logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            # logger.info(f"skipped: {skipped / 2 ** 20}M params")

    num_warmup_steps = args.warmup_steps if args.warmup_steps >= 1 else estimated_stepping_batches * args.warmup_steps
    num_warmup_steps = int(num_warmup_steps)
    lr_scheduler_type = args.scheduler_type.lower()


    scheduler = None
    if lr_scheduler_type == 'WarmupCosine'.lower():
        lr_scheduler_type = SchedulerType.COSINE
    elif lr_scheduler_type == 'CosineAnnealingWarmRestarts'.lower():
        lr_scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
    elif lr_scheduler_type in ['CAL'.lower(), 'CosineAnnealingLR'.lower()]:
        T_mult = args.scheduler["T_mult"]
        eta_min = args.scheduler.get('eta_min', 0.)
        last_epoch = args.scheduler.get('last_epoch', -1)
        verbose = args.scheduler.get('verbose', False)
        if args.scheduler.get('T_0', None) is None:
            rewarm_epoch_num = args.scheduler["rewarm_epoch_num"]
            T_0 = int(estimated_stepping_batches * rewarm_epoch_num / args.max_epochs)
            T_0 = max(T_0, 1)
        else:
            T_0 = int(args.scheduler["T_0"])
            T_0 = max(T_0, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult,
                                                                   eta_min=eta_min,
                                                                   last_epoch=last_epoch,
                                                                   verbose=verbose)
    elif lr_scheduler_type in ['CAWR'.lower(), 'CosineAnnealingWarmRestarts'.lower()]:
        # T_mult = args.scheduler["T_mult"]
        eta_min = args.scheduler.get('eta_min', 0.)
        last_epoch = args.scheduler.get('last_epoch', -1)
        verbose = args.scheduler.get('verbose', False)
        if args.scheduler.get('T_0', None) is None:
            rewarm_epoch_num = args.scheduler["rewarm_epoch_num"]
            T_0 = int(estimated_stepping_batches * rewarm_epoch_num / args.max_epochs)
            T_0 = max(T_0, 1)
        else:
            T_0 = int(args.scheduler["T_0"])
            T_0 = max(T_0, 1)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_0,
                                                         eta_min=eta_min,
                                                         last_epoch=last_epoch,
                                                         verbose=verbose)
    elif args.scheduler_type.lower() == 'Step'.lower():
        decay_rate = args.scheduler["decay_rate"]
        decay_steps = args.scheduler["decay_steps"]
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    elif args.scheduler_type.lower() == 'ReduceLROnPlateau'.lower():
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)
    elif args.scheduler_type.lower() == 'WarmupCosine'.lower():
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=estimated_stepping_batches)

    if scheduler is None:
        scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    if scheduler:
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", "frequency": 1
            },
        }
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


def get_value_from_args_assert(key,dtype,*args,**kwargs):
    value = kwargs.get(key, None)
    if value is not None:
        for item in args:
            if isinstance(item,dtype):
                value = item
                break
    assert value is not None, ValueError('no param ',key)
    return value

def get_value_from_args(key,dtype,*args,**kwargs):
    value = kwargs.get(key, None)
    if value is not None:
        for item in args:
            if isinstance(item,dtype):
                value = item
                break
    return value