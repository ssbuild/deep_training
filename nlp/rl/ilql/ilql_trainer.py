# coding=utf8
# @Time    : 2023/5/3 14:19
# @Author  : tk
# @FileName: ilql_trainner
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
from torch.utils.data import DataLoader
from lightning_utilities import is_overridden
from lightning_utilities.core.apply_func import apply_to_collection
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects, _FabricModule

from .ilql_dataset import ILQLSeq2SeqRolloutStorage, ILQLRolloutStorage, tokenize_dialogue
from ..rl_base.rl_dataset import MiniBatchIterator, logger
from ..utils import RunningMoments
from .configuration import ILQLConfig

from lightning.fabric.loggers.tensorboard import TensorBoardLogger


class ILQLTrainer:
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
        self._state: Optional[dict] = {
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


    #model,tokenizer,reward_fn,ilql_config
    def prepare_fit(self, model: L.LightningModule,
        tokenizer,
        ilql_config,
        reward_fn = None,
        **kwargs):

        self.config = model.config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ilql_config: ILQLConfig = ilql_config

        # Setup stats tracker
        self.running_moments = RunningMoments()

        self.train_mb_count = 0
        self.train_item_count = 0

        if self.ilql_config.minibatch_size:
            assert model.training_args.train_batch_size % self.ilql_config.minibatch_size == 0, "Minibatch size must divide batch size"
            self.mb_size = self.ilql_config.minibatch_size
        else:
            self.mb_size = model.training_args.train_batch_size




    def fit(
        self,
        model: L.LightningModule,
        train_loader: DataLoader,
        tokenizer,
        ilql_config,
        reward_fn = None,
        val_loader: Optional[DataLoader] = None,
        ckpt_path: Optional[str] = None,
        stop_sequences = None,
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

        self.prepare_fit(model,tokenizer,ilql_config,reward_fn)
        self.stop_sequences = stop_sequences

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader,
                                                     use_distributed_sampler=self.use_distributed_sampler,
                                                     move_to_device=False)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler,
                                                       move_to_device=False)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")
        else:
            optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
            assert optimizer is not None
            model, optimizer = self.fabric.setup(model, optimizer)

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



        while not self.should_stop:

            self.train_loop(
                model, optimizer,train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
            )

            if self.should_validate:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # reset for next fit call
        self.should_stop = False

    def post_backward_callback(self,model: _FabricModule):
        if self.train_item_count % self.ilql_config.steps_for_target_q_sync == 0:
            model.module.backbone.sync_target_q_heads()

    def post_epoch_callback(self,*agrs,**kwargs):
        ...

    @property
    def callback_metrics(self):
        return self._callback_metrics

    def train_loop(
        self,
        model: _FabricModule,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
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
        self.fabric.call("on_train_epoch_start",self,model)
        iterable = self.progbar_wrapper(
            train_loader, total=len(train_loader), desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_train_epoch_end",self,model)
                return

            self.fabric.call("on_train_batch_start",self,model,batch, batch_idx)
            # Note that whereas standard policy gradient methods perform one
            # gradient update per batch, PPO for example commonly performs
            # multiple gradient updates on the same batch of data.
            # https://arxiv.org/pdf/1707.06347.pdf
            stats_accum = []
            loss_accum = []

            bs = batch['input_ids'].size(0)
            num_mb = bs // self.mb_size
            if num_mb == 0:
                num_mb = 1
                mbs = [batch]
            else:
                batch_keys = batch.keys()
                mbs = []
                for i in range(num_mb):
                    mb = {k: None for k in batch_keys}
                    for k in batch_keys:
                        mb[k] = batch[k][i * self.mb_size: (i + 1) * self.mb_size]
                    mbs.append(mb)

            for mb in mbs:
                self.train_mb_count += 1
                for k in mb:
                    mb[k] = mb[k].to(self.fabric.device)
                should_sync = self.train_mb_count % self.accumulate_grad_batches == 0
                with self.fabric.no_backward_sync(model,enabled=not should_sync):
                    outputs = self.training_step(model=model, batch=mb, batch_idx = batch_idx)
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
        self.post_epoch_callback(model=model)


    def val_loop(
        self,
        model: _FabricModule,
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
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch)
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




    def make_experience(self, samples, rewards, max_length=2048, verbose=True):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        if verbose:
            logger.info("Collecting rollouts")

        if self.ilql_config.model_arch_type == "seq2seq":
            self.store = self.make_experience_seq2seq(samples, rewards, max_length)
        else:
            self.store = self.make_causal_experience(samples, rewards, self.tokenizer, max_length=max_length)

    def make_experience_seq2seq(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_output_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            all_input_ids.append(torch.tensor(sample[0].tokens))
            all_output_ids.append(torch.tensor(sample[1].tokens))
            actions_ixs = []
            length = 0
            for phrase in sample:
                if phrase.is_output:
                    length = len(phrase.tokens)
                    actions_ixs.append(torch.arange(0, length - 1))
            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=torch.int32))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        # if self.tokenizer and os.environ.get("RANK", "0") == "0":
        #     logger.info("Logging sample example")
        #     prompt = self.tokenizer.decode(all_input_ids[0])
        #     response = self.tokenizer.decode(all_output_ids[0])
        #     columns = ["Prompt", "Response", "Reward"]
        #     table = Table(*columns, title="Sample Example", show_lines=True)
        #     table.add_row(prompt, response, str(rewards[0]))
        #     Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids))) + np.array(list(map(len, all_output_ids)))
        output_lengths = np.array(list(map(len, all_output_ids)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=torch.float32)

        # if os.environ.get("RANK", "0") == "0":
        #     logger.info("Logging experience string statistics")
        #     columns = ["Prompt Length", "Output Length", "Sample Length"]
        #     table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
        #     row = []
        #     for lengths in [prompt_lengths, output_lengths, sample_lengths]:
        #         row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
        #     table.add_row(*row)
        #     Console().print(table)

        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(returns.dtype).eps)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=torch.int32) for x in all_input_ids]
        return ILQLSeq2SeqRolloutStorage(
            all_input_ids,
            attention_mask,
            all_output_ids,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )

    def make_causal_experience(self,samples, rewards, tokenizer=None, max_length=2048):  # noqa: C901
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        if tokenizer is not None:
            samples = [tokenize_dialogue(s, tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            length = 0
            all_input_ids.append(torch.tensor(sum((s.tokens for s in sample), ())))
            actions_ixs = []
            for dm in sample:
                if dm.is_output:
                    actions_ixs.append(torch.arange(length - 1, length + len(dm.tokens) - 1))

                length += len(dm.tokens)

            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=torch.int32))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        # if tokenizer is not None and os.environ.get("RANK", "0") == "0" and verbose:
        #     logger.info("Logging sample example")
        #     prompt = tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
        #     response = tokenizer.decode(all_input_ids[0][all_states_ixs[0][1]:])
        #     columns = ["Prompt", "Response", "Reward"]
        #     table = Table(*columns, title="Sample Example", show_lines=True)
        #     table.add_row(prompt, response, str(rewards[0]))
        #     Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids)))
        output_lengths = np.array(list(map(len, all_actions_ixs)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=torch.float32)

        # if os.environ.get("RANK", "0") == "0" and verbose:
        #     logger.info("Logging experience string statistics")
        #     columns = ["Prompt Length", "Output Length", "Sample Length"]
        #     table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
        #     row = []
        #     for lengths in [prompt_lengths, output_lengths, sample_lengths]:
        #         row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
        #     table.add_row(*row)
        #     Console().print(table)

        returns = returns - returns.mean()
        std_returns = returns.std()
        if not torch.isnan(std_returns):
            returns = returns / (std_returns + torch.finfo(returns.dtype).eps)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=torch.int32) for x in all_input_ids]

        return ILQLRolloutStorage(
            all_input_ids,
            attention_mask,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )
