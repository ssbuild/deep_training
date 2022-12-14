# @Time    : 2022/11/17 22:18
# @Author  : tk
# @FileName: training_args.py
import os
from dataclasses import dataclass, field
from typing import Optional
from pytorch_lightning.utilities.seed import seed_everything

__all__ = [
    'ModelArguments',
    'PrefixModelArguments',
    'TrainingArguments',
    'DataArguments',
    'MlmDataArguments',
]

@dataclass
class ModelArguments:
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
        default=True,
        metadata={"help": "Whether to lower case the input text. Should be True for uncased deep_training and False for cased deep_training."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
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
class PrefixModelArguments:
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
class TrainingArguments:
    optimizer: str = field(
        default='adamw',
        metadata={"help": "one of [adamw,adam]"},
    )
    scheduler_type: str = field(
        default='linear',
        metadata={"help": "one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau]"},
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

    learning_rate : float = field(
        default=5e-5,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    learning_rate_for_task: float = field(
        default=5e-5,
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
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam优化器的epsilon值"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "梯度积累"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "模型任务层训练时的学习率"},
    )

    warmup_steps: int = field(
        default=0,
        metadata={"help": "模型任务层训练时的学习率"},
    )

    train_batch_size: int = field(
        default=16,
        metadata={"help": "模型任务层训练时的学习率"},
    )

    eval_batch_size: int = field(
        default=1,
        metadata={"help": "模型任务层训练时的学习率"},
    )

    test_batch_size: int = field(
        default=1,
        metadata={"help": "模型任务层训练时的学习率"},
    )
    seed: float = field(
        default=42,
        metadata={"help": "模型任务层训练时的学习率"},
    )

    def __post_init__(self):
        seed_everything(self.seed)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    devices: Optional[int] = field(
        default="1",metadata={
            "help": "device str"
        }
    )
    data_backend: Optional[str] = field(
        default=None, metadata={"help": "record,leveldb,lmdb,memory,memory_raw"}
    )
    convert_file: Optional[bool] = field(
        default=True, metadata={"help": "是否需要转换语料到record记录"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "训练语料，多个文件“,”分割"}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "评估语料，多个文件“,”分割"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "测试语料，多个文件“,”分割"}
    )
    label_file: Optional[str] = field(
        default=None, metadata={"help": "标签文件，多个文件“,”分割"}
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
        if self.train_file:
            self.train_file = self.train_file.split(',')

        if self.eval_file:
            self.eval_file = self.eval_file.split(',')

        if self.test_file:
            self.test_file = self.test_file.split(',')


        if self.label_file:
            self.label_file = self.label_file.split(',')

        if not self.train_file:
            self.do_train = False

        if not self.eval_file:
            self.do_eval = False

        if not self.test_file:
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
class MlmDataArguments:
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

