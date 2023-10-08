# @Time    : 2022/11/17 22:18
# @Author  : tk
# @FileName: training_args.py
import copy
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional,Union,List,Dict,Any
import torch
from packaging import version
from transformers import TrainingArguments as TrainingArgumentsHF_, IntervalStrategy, is_torch_available

from transformers.trainer_utils import EvaluationStrategy
from transformers.training_args import default_logdir
from transformers.utils import logging
from ..nlp.optimizer.optimizer import OptimizerNames
from ..nlp.scheduler.scheduler import SchedulerType
from ..utils.function import copy_dataclass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

__all__ = [
    'ModelArguments',
    'PrefixModelArguments',
    'TrainingArguments',
    'TrainingArgumentsHF',
    'TrainingArgumentsCL',
    'DataArguments',
    'MlmDataArguments',

]

class _ArgumentsBase:
    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__
    def __deepcopy__(self, memodict={}):
        return self.__class__(**copy.deepcopy(asdict(self)))



@dataclass
class TrainingArgumentsHF(TrainingArgumentsHF_,_ArgumentsBase):

    data_backend: Optional[str] = field(
        default="record",
        metadata={
            "help": (
                "default data_backend."
            )
        },
    )
    learning_rate_for_task: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "learning_rate_for_task."
            )
        },
    )


    def __post_init__(self):
        super().__post_init__()
        if self.learning_rate_for_task is None:
            self.learning_rate_for_task = self.learning_rate

@dataclass
class TrainingArgumentsCL(_ArgumentsBase):

    framework = "pt"
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    data_backend: Optional[ str ] = field(
        default="record",
        metadata={
            "help": (
                "default data_backend."
            )
        },
    )
    strategy: Optional[ Union[str,Dict ]] = field(
        default="gemini",
        metadata={
            "help": (
                "one of . ddp " "gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"
            )
        },
    )

    learning_rate_for_task: Optional[ float ] = field(
        default=None,
        metadata={
            "help": (
                "learning_rate_for_task."
            )
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: Union[ IntervalStrategy, str ] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[ int ] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    eval_delay: Optional[ float ] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: Union[ SchedulerType, str ] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})


    logging_dir: Optional[ str ] = field(default=None, metadata={"help": "Tensorboard log dir."})

    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )

    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: Union[ IntervalStrategy, str ] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[ int ] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_safetensors: Optional[ bool ] = field(
        default=False,
        metadata={
            "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers."},
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": " Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available."
        },
    )

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[ int ] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )


    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: Optional[ bool ] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})


    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: Optional[ float ] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )



    run_name: Optional[ str ] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[ bool ] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[ bool ] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[ List[ str ] ] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    load_best_model_at_end: Optional[ bool ] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    metric_for_best_model: Optional[ str ] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[ bool ] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )


    default_optim = "adamw_torch"
    # XXX: enable when pytorch==2.0.1 comes out - we want to give it time to get all the bugs sorted out
    # if is_torch_available() and version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.1.0"):
    #     default_optim = "adamw_torch_fused"
    # and update the doc above to:
    # optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch_fused"` (for torch<2.1.0 `"adamw_torch"`):
    optim: Union[ OptimizerNames, str ] = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    optim_args: Optional[ str ] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[ str ] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )


    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: Optional[ str ] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=False, metadata={"help": "Whether or not the inputs will be passed to the `compute_metrics` function."}
    )

    include_tokens_per_second: Optional[ bool ] = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )


    def __post_init__(self):
        if self.learning_rate_for_task is None:
            self.learning_rate_for_task = self.learning_rate

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5"
                " of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value
        if self.no_cuda:
            warnings.warn(
                "using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. "
                "Use `use_cpu` instead",
                FutureWarning,
            )
            self.use_cpu = self.no_cuda

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps"
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    # Work around floating point precision issues
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )


        if (
                self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir


        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(
                    " `--half_precision_backend apex`: GPU bf16 is not supported by apex. Use"
                    " `--half_precision_backend cuda_amp` instead"
                )

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.evaluation_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_torch_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0")

        self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim"
                " adafactor` instead",
                FutureWarning,
            )
            self.optim = OptimizerNames.ADAFACTOR
        if self.optim == OptimizerNames.ADAMW_TORCH_FUSED and is_torch_available():
            if version.parse(version.parse(torch.__version__).base_version) < version.parse("2.0.0"):
                raise ValueError("--optim adamw_torch_fused requires PyTorch 2.0 or higher")
            # there is a bug in fp16/AMP in pt-2.0.0
            if version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.0") and self.fp16:
                raise ValueError("--optim adamw_torch_fused with --fp16 requires PyTorch>2.0")

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps



@dataclass
class ModelArguments(_ArgumentsBase):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: bool = field(
        default=None,
        metadata={"help": "Whether to lower case the input text. Should be True for uncased deep_training and False for cased deep_training."},
    )
    use_fast_tokenizer: bool = field(
        default=None,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class PrefixModelArguments(_ArgumentsBase):
    # prompt参数
    prompt_type: int = field(
        default=0,
        metadata={
            "help": "0 : prefix model , 1 prompt model"
        }
    )

    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "prefix_projection"
        }
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used'"
        }
    )
    pre_seq_len: int = field(
        default=16,
        metadata={
            "help": "The length of prompt"
        }
    )



@dataclass
class TrainingArguments(_ArgumentsBase):

    optimizer: str = field(
        default='adamw',
        metadata={"help": "one of lamb,adam,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,"
                          "adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,"
                          "paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,"
                          "lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp"},
    )
    optimizer_args: Optional[str] = field(default=None,metadata={"help": "sample a=100,b=10 "})
    scheduler_type: str = field(
        default='linear',
        metadata={"help": "one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, "
                          "cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]"},
    )

    scheduler: dict = field(
        default=None,
        # {
        #     # StepLR
        #     "decay_rate": 0.999,
        #     "decay_steps": 100,
        # }

        # {
        #     # CosineAnnealingWarmRestarts
        #     "T_mult": 1,
        #     "rewarm_epoch_num": 2,
        # }
        metadata={"help": "StepLR:  { 'decay_rate': 0.999,'decay_steps': 100,'verbose': True} ,\
                          CAWR {'T_mult': 1, 'rewarm_epoch_num': 2,'verbose': True} ,\
                          CAL: {'rewarm_epoch_num': 2,'verbose': True} \
                          "},
    )
    adv: dict = field(
        # default_factory= lambda: {
        #     'mode': None, # None, fgm, fgsm_local, fgsm, pgd, free_local, free
        #     'emb_name=': 'embedding',
        #     'attack_iters': 2, # pgd
        #     'minibatch_replays': 2, # free
        #     'alpha': 0.1, # pgd
        #     'epsilon': 1.0 # pgd,fgm
        # },
        default=None,
        metadata={"help": "对抗训练"},
    )
    hierarchical_position: float = field(
        default=None,
        metadata={"help": "层次分解位置编码，让transformer可以处理超长文本 , 绝对位置编码有效 , None禁用 , 0 - 1 启用 "},
    )

    learning_rate : float = field(
        default=5e-5,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    learning_rate_for_task: float = field(
        default=None,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    max_epochs: int = field(
        default=-1,
        metadata={"help": "模型训练的轮数"},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "max_steps"},
    )
    optimizer_betas : tuple = field (
        default=(0.9, 0.999),
        metadata={"help": "优化器的betas值"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam优化器的epsilon值"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "gradient_accumulation_steps"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "max_grad_norm"},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "weight_decay"},
    )

    warmup_steps: float = field(
        default=0,
        metadata={"help": "warmup_steps"},
    )

    train_batch_size: int = field(
        default=16,
        metadata={"help": "train_batch_size"},
    )

    eval_batch_size: int = field(
        default=1,
        metadata={"help": "eval_batch_size"},
    )

    test_batch_size: int = field(
        default=1,
        metadata={"help": "test_batch_size"},
    )
    seed: Optional[float] = field(
        default=42,
        metadata={"help": "seed"},
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    torch_compile: bool = field(
        default=False, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    def __post_init__(self):
        if self.learning_rate_for_task is None:
            self.learning_rate_for_task = self.learning_rate

        if self.seed is not None:
            from lightning_fabric.utilities.seed import seed_everything
            seed_everything(int(self.seed))


        assert self.hierarchical_position is None or (self.hierarchical_position >0 and self.hierarchical_position <1)


@dataclass
class DataArguments(_ArgumentsBase):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    devices: Optional[int] = field(
        default="1",metadata={
            "help": "device str"
        }
    )
    convert_onnx: Optional[bool] =  field(
        default=False, metadata={"help": "是否转换onnx"}
    )
    data_backend: Optional[str] = field(
        default=None, metadata={"help": "record,leveldb,lmdb,memory,memory_raw"}
    )
    convert_file: Optional[bool] = field(
        default=True, metadata={"help": "是否需要转换语料到record记录"}
    )
    train_file: Optional = field(
        default_factory=lambda: [], metadata={"help": "训练语料list"}
    )
    eval_file: Optional = field(
        default_factory=lambda: [], metadata={"help": "评估语料list"}
    )
    test_file: Optional = field(
        default_factory=lambda: [],metadata={"help": "测试语料list"}
    )
    label_file: Optional = field(
        default_factory=lambda: [], metadata={"help": "标签文件list"}
    )
    intermediate_name: Optional[str] = field(
        default='dataset', metadata={"help": "dataset文件名前缀"}
    )
    output_dir: Optional[str] = field(
        default='./output', metadata={"help": "模型输出路径"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    eval_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    test_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "语言生成标题的最大长度 "
            )
        },
    )
    do_train: bool = field(
        default=False, metadata={"help": "是否训练"}
    )
    do_eval: bool = field(
        default=False, metadata={"help": "是否评估"}
    )
    do_test: bool = field(
        default=False, metadata={"help": "是否测试"}
    )

    def __post_init__(self):

        if not self.train_file:
            self.do_train = False

        if not self.eval_file:
            self.do_eval = False

        if not self.test_file:
            self.do_test = False

        if self.convert_onnx:
            self.do_train = False
            self.do_eval = False
            self.do_test = False



        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if self.train_max_seq_length is None:
            self.train_max_seq_length = self.max_seq_length
        if self.eval_max_seq_length is None:
            self.eval_max_seq_length = self.max_seq_length
        if self.test_max_seq_length is None:
            self.test_max_seq_length = self.max_seq_length

@dataclass
class MlmDataArguments(_ArgumentsBase):
    do_whole_word_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to use whole word masking rather than per-WordPiece masking."
        }
    )
    max_predictions_per_seq: int = field(
        default=20,
        metadata={
            "help": "Maximum number of masked LM predictions per sequence."
        }
    )
    masked_lm_prob: float = field(
        default=0.15,
        metadata={
            "help": "Masked LM probability."
        }
    )
    dupe_factor: int = field(
        default=5,
        metadata={
            "help": "Number of times to duplicate the input data (with different masks)."
        }
    )

