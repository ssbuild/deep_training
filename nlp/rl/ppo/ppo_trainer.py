# coding=utf8
# @Time    : 2023/5/3 14:19
# @Author  : tk
# @FileName: ppo_trainner
import typing
import os

import lightning
import numpy as np
from tqdm import tqdm
from time import time
from collections.abc import Mapping
from functools import partial
from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union, Callable
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning_utilities import is_overridden
from lightning_utilities.core.apply_func import apply_to_collection
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects, _FabricModule
from .ppo_dataset import PPORolloutStore
from .data_define import PPORLElement,logger,logging
from ..rl_base.rl_dataset import MiniBatchIterator
from ..utils import logprobs_of_labels, Clock, gather_dict, RunningMoments, pad_across_processes, _gpu_gather, infinite_dataloader
from ...layers.ppo import AdaptiveKLController, FixedKLController
from .configuration import PPOConfig

from lightning.fabric.loggers.tensorboard import TensorBoardLogger


class PPOTrainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        accumulate_grad_batches: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        max_grad_norm=None,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            accumulate_grad_batches: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!
        """

        if loggers is None:
            loggers = TensorBoardLogger(root_dir=checkpoint_dir,name='lightning_logs')

        self.loggers = loggers
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.global_step = 0
        self.accumulate_grad_batches: int = accumulate_grad_batches
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.max_grad_norm = max_grad_norm
        self.train_mb_count = 0
        self.train_item_count = 0

        self._callback_metrics: dict = {}
        self._state : Optional[dict] = {
            "pytorch-lightning_version": lightning.__version__
        }
        self.fabric.launch()


    @property
    def world_size(self):
        return self.fabric.world_size

    @property
    def global_rank(self):
        return self.fabric.global_rank

    @property
    def local_rank(self):
        return self.fabric.local_rank



    #model,tokenizer,reward_fn,ppo_config,stop_sequences
    def prepare_fit(self, model: L.LightningModule,
        tokenizer,
        reward_fn: Callable,
        ppo_config,
        stop_sequences=None,**kwargs):

        self.config = model.config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ppo_config: PPOConfig = ppo_config

        self.stop_sequences = stop_sequences

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.ppo_config.ref_mean
        self.ref_std = self.ppo_config.ref_std

        self.store = PPORolloutStore(self.tokenizer.pad_token_id, self.tokenizer.padding_side)
        self.train_mb_count = 0
        self.train_item_count = 0

        if self.ppo_config.minibatch_size:
            assert model.training_args.train_batch_size % self.ppo_config.minibatch_size == 0, "Minibatch size must divide batch size"
            self.mb_size = self.ppo_config.minibatch_size
        else:
            self.mb_size = model.training_args.train_batch_size

        # self.fabric.barrier()
        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        if self.ppo_config.target is not None:
            self.kl_ctl = AdaptiveKLController(self.ppo_config.init_kl_coef, self.ppo_config.target,
                                               self.ppo_config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.ppo_config.init_kl_coef)
        
        if self.ppo_config.model_arch_type == "seq2seq":
            self.generate_kwargs = dict(
                self.ppo_config.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.ppo_config.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    self.ppo_config.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        else:
            self.generate_kwargs = dict(
                self.ppo_config.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if self.ppo_config.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    self.ppo_config.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                self.generate_experience_kwargs = None
                
        self.n_updates_per_batch = self.ppo_config.ppo_epochs

    def fit(
        self,
        model: L.LightningModule,
        ref_model: Optional[L.LightningModule],
        train_loader: DataLoader,
        tokenizer,
        reward_fn: Callable,
        ppo_config,
        val_loader: Optional[DataLoader] = None,
        stop_sequences=None,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`PPPTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
        """

        self.prepare_fit(model,tokenizer,reward_fn,ppo_config,stop_sequences)

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader,
                                                     use_distributed_sampler=self.use_distributed_sampler,
                                                     move_to_device=True)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader,
                                                       use_distributed_sampler=self.use_distributed_sampler,
                                                       move_to_device=True)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")
        else:
            optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
            assert optimizer is not None
            model, optimizer = self.fabric.setup(model, optimizer)
            # ref_model = self.fabric.setup(ref_model)

        # assemble state (current epoch and global step will be added in save)
        state = {"state_dict": model, "optimizer_states": optimizer, "lr_schedulers": scheduler_cfg}
        self._state.update(state)
        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        ref_model = ref_model.to(model.device)
        self.prompt_train_loader: typing.Iterator = infinite_dataloader(train_loader)
        self.make_experience(model,ref_model)
        while not self.should_stop:

            self.train_loop(
                model, ref_model, optimizer, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
            )

            if self.should_validate:
                self.val_loop(model,ref_model, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # reset for next fit call
        self.should_stop = False

    def post_backward_callback(self,model):
        self.kl_ctl.update(self.mean_kl, n_steps=model.training_args.train_batch_size)

    def post_epoch_callback(self,*agrs,**kwargs):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if not self.should_stop:
            model = kwargs['model']
            ref_model = kwargs['ref_model']
            self.store.clear_history()
            # Collect more rollouts for training
            self.make_experience(model, ref_model)

    @property
    def callback_metrics(self):
        return self._callback_metrics

    def train_loop(
        self,
        model: _FabricModule,
        ref_model: Optional[_FabricModule],
        optimizer: torch.optim.Optimizer,
        # train_loader: DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater then the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightninModule.configure_optimizers` for supported values.
        """


        num_mb = model.training_args.train_batch_size // self.mb_size
        train_loader = self.fabric.setup_dataloaders(self.store.create_loader(model.training_args.train_batch_size,shuffle=True),
                                                     use_distributed_sampler=False)
        mbs = MiniBatchIterator(train_loader, self.mb_size, num_mb)
        self.fabric.call("on_train_epoch_start",self,model)
        iterable = self.progbar_wrapper(
            mbs, total=len(train_loader) * num_mb , desc=f"Epoch {self.current_epoch}"
        )
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_train_epoch_end",self,model)
                return

            self.fabric.call("on_train_batch_start",self,model,mbs, batch_idx)
            # For each update per batch
            for _ in range(self.n_updates_per_batch):
                # Note that whereas standard policy gradient methods perform one
                # gradient update per batch, PPO for example commonly performs
                # multiple gradient updates on the same batch of data.
                # https://arxiv.org/pdf/1707.06347.pdf
                stats_accum = []
                loss_accum = []
                for mb in batch:
                    self.train_mb_count += 1
                    should_sync = self.train_mb_count % self.accumulate_grad_batches == 0
                    with self.fabric.no_backward_sync(model,enabled=not should_sync):
                        outputs = self.training_step(model=model, batch = mb, batch_idx = batch_idx)
                        loss, stats = outputs['loss'], outputs['stats']
                    loss_accum.append(loss)
                    stats_accum.append(stats)

                # TODO(Dahoas): Best way to combine stats between mbs?
                # How does accelerate do it?
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step" ,self,model,optimizer, 0)

                if self.max_grad_norm is not None:
                    self.fabric.clip_gradients(model, optimizer, max_norm=self.max_grad_norm)
                # optimizer step runs train step internally through closure
                optimizer.step()
                self.fabric.call("on_before_zero_grad",self,model, optimizer)
                optimizer.zero_grad()

                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)


                # only increase global step if optimizer stepped
                self.global_step += 1

                self.train_item_count += 1
                stats = {key: sum([stats[key] for stats in stats_accum]) / num_mb for key in stats_accum[0]}
                metrics = {
                    "loss": torch.mean(torch.stack(loss_accum)),
                }
                metrics.update(stats)
                self.fabric.logger.log_metrics(metrics, step=self.global_step)
                self._callback_metrics.update(metrics)

                self.fabric.call("on_train_batch_end", self, model, self._current_train_return, batch, batch_idx)

                # stopping criterion on step level
                if  self.max_steps is not None and self.max_steps >= 0 and self.global_step >= self.max_steps:
                    self.should_stop = True
                    break

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return['loss'] ,"train")
            self.post_backward_callback(model)
        self.fabric.call("on_train_epoch_end",self,model)
        self.post_epoch_callback(model=model,ref_model=ref_model)


    def val_loop(
        self,
        model: _FabricModule,
        ref_model: Optional[_FabricModule],
        val_loader: Optional[DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater then the number of batches in the ``val_loader``, this has no effect.
        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        elif val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model), L.LightningModule):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval",self,model)  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start",self,model)

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_validation_epoch_end",self,model)
                return

            self.fabric.call("on_validation_batch_start",self,model, batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end",self,model, out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

        self.fabric.call("on_validation_epoch_end",self,model)

        self.fabric.call("on_validation_model_train",self,model)
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this
        is given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        forward_time,backward_time = 0,0
        forward_time -= time()
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch,device=self.fabric.device)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        forward_time += time()

        self.fabric.call("on_before_backward",self,model, loss)
        backward_time -= time()
        self.fabric.backward(loss)
        backward_time += time()
        self.fabric.call("on_after_backward",self,model)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        stats = self._current_train_return['stats']
        stats['time/forward'] = forward_time
        stats['time/backward'] = backward_time
        return self._current_train_return

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightninModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``
        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from
        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save_checkpoint(self, filepath,weights_only) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """
        state = self._state
        state.update(global_step=self.global_step, current_epoch=self.current_epoch)
        if weights_only:
            state = {
                "state_dict": state['state_dict'],
                "pytorch-lightning_version": state['pytorch-lightning_version'],
            }
        self.fabric.save(filepath, state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints
        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.
        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        elif isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        elif isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        elif isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            elif all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    if isinstance(v, Mapping):
                        for sub_k,sub_v in v.items():
                            postfix_str += f" {prefix}_{k}/{sub_k}: {sub_v:.3f}"
                    else:
                        postfix_str += f" {prefix}_{k}: {v:.3f}"
            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)
        return model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    #
    def decode(
            self,
            prompts: List[torch.LongTensor],
            samples: List[torch.LongTensor],
            prompt_sizes: torch.LongTensor = None,
            append_eos_token: bool = False,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            if self.ppo_config.model_arch_type == "seq2seq":
                output_start_ix = 0
            else:
                output_start_ix = prompt_size

            str_prompt = self.tokenizer.decode(prompt[:prompt_size], skip_special_tokens=True)
            str_output = self.tokenizer.decode(sample[output_start_ix:], skip_special_tokens=True)
            # Trim outputs up to `self.stop_sequences` if any are present
            trimmed = False
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()
                        trimmed = True

            # Recover the last <eos> if it was present in the original sample
            # or add one if it was trimmed with `self.stop_sequences`.
            # Only in cases when a generation ended due to `max_new_tokens` exhaustion,
            # <eos> token would not be present in the original sample
            if append_eos_token and (trimmed or sample[-1] == self.tokenizer.eos_token_id):
                str_output += self.tokenizer.eos_token

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)



            if self.ppo_config.model_arch_type == "seq2seq":
                if hasattr(self.tokenizer,'_sep_token') and self.tokenizer._sep_token is not None:
                    sample = str_prompt + self.tokenizer.sep_token + str_output
                else:
                    sample = str_prompt + str_output
            elif self.ppo_config.model_arch_type == "prefixlm":
                sample = str_prompt + self.tokenizer.gmask_token + self.tokenizer.bos_token + str_output
            else:
                sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs

    def make_experience(self, model, ref_model,**kwargs):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        num_rollouts = self.ppo_config.num_rollouts
        world_size = self.fabric.world_size
        is_main_process = self.fabric.is_global_zero
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=not is_main_process,
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        self.store.clear_history()

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []
        prompt_iterator : typing.Iterator = self.prompt_train_loader
        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: dict = next(prompt_iterator)

            rollout_generate_time = time()

            if self.ppo_config.model_arch_type == "prefixlm":
                attention_mask = None
            else:
                attention_mask = batch.get('attention_mask', None)
            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(model , batch["input_ids"], attention_mask , **kwargs)
            stats["rollout/time/generate"] = time() - rollout_generate_time
            prompt_tensors = batch['input_ids']
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = pad_across_processes(
                samples,world_size, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = pad_across_processes(
                prompt_tensors, world_size,dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = _gpu_gather(padded_samples,world_size)
            gathered_prompts = _gpu_gather(padded_prompts,world_size)
            gathered_prompt_sizes = _gpu_gather(prompt_sizes,world_size)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask" and k != "position_ids"})


            if is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                rollout_score_time = time()
                all_scores = self.reward_fn(
                    samples=all_str_samples, prompts=all_str_prompts, outputs=all_str_outputs, **metadata
                )
                all_scores = all_scores.clone().detach().float().to(device)
                stats["rollout/time/score"] = time() - rollout_score_time
                all_scores = list(all_scores.reshape(world_size, -1).unbind())
            else:
                all_scores = None

            if dist.is_initialized():
                scores = torch.empty(len(samples), device=device)
                dist.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.ppo_config.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.ppo_config.cliprange_reward:
                scores = torch.clip(scores, -self.ppo_config.cliprange_reward, self.ppo_config.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["rollout/scores/mean"] = all_scores_mean.item()
            stats["rollout/scores/std"] = all_scores_std.item()
            stats["rollout/scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout/scores/running_std"] = self.running_moments.std.item()

            if self.ppo_config.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.ppo_config.scale_reward == "ref":
                scores /= self.ref_std

            # Precompute logprobs, values
            if self.ppo_config.model_arch_type == "seq2seq":
                attention_mask = batch['attention_mask'].to(device)
                prompt_tensors = batch['input_ids'].to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = model.forward_logits_values(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                        return_dict=True,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(model, "frozen_head"):
                        ref_logits = model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = ref_model.forward_logits_values(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                if self.ppo_config.model_arch_type == "causal":
                    attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                else:
                    attention_mask = None
                with torch.no_grad():
                    logits, *_, values = model.forward_logits_values(
                        input_ids=all_tokens,
                        attention_mask=attention_mask,
                    )

                    ref_logits = ref_model.forward_logits_values(
                        input_ids=all_tokens,
                        attention_mask=attention_mask,
                        return_dict=True,
                    ).logits
                    ref_logits = ref_logits.to(device)
                if attention_mask is None:
                    attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)

            if self.ppo_config.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.ppo_config.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1


            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the start of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start: ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start: ends[ix]] for ix in range(n_samples)]

            kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            kl_penalty = [xs[start: ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0

            for sample_idx in range(n_samples):
                rewards = kl_penalty[sample_idx]
                rewards[-1] += scores[sample_idx].cpu()

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if dist.is_initialized():
                dist.all_reduce(mean_kl, dist.ReduceOp.AVG)

            stats["rollout/time"] = clock.tick()
            stats["rollout/policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["rollout/policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["rollout/kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["rollout/policy/sqrt_kl"] ** 2

        self.fabric.logger.log_metrics(stats,self.global_step)

        # Push samples and rewards to trainer's rollout storage
        self.store.push(ppo_rl_elements)