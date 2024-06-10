from dataclasses import dataclass, field, fields
from typing import Optional

@dataclass
class DataTrainingArguments:
    data_dir: Optional[str] = field(
        default='./data/',
        metadata={
            "help": "The data directory"
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The train file"
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The validation file"
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "Max sequence length"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    translation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The translated (from eng) file"
        },
    )
    sts_source: Optional[str] = field(
        default='stsb',
        metadata={
            "help": "The sts source",
        },
    )
    mix_sts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Mix STR with STS"
        },
    )
    max_num : Optional[int] = field(
        default=-1,
        metadata={
            "help": "The maximum number of examples to keep for nli/sts training. -1 means all examples"
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained model name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer name"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class SemArguments:
    type: Optional[str] = field(
        default = 'cross',
        metadata={
            "choices": ["bi", "cross"],
            "help": "The type of training"
        },
    )
    language: Optional[str] = field(
        default = 'eng',
        metadata={
            "help": "The language for training and evaluation"
        },
    )

@dataclass
class WandbArguments:
    disable_wandb: bool = field(
        default=False, metadata={"help": "Whether to disable wandb logging."}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "The name of the wandb project."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the current run."}
    )
    wandb_group: Optional[str] = field(
        default=None, metadata={"help": "The group name for the current run."}
    )
    wandb_job_type: Optional[str] = field(
        default=None, metadata={"help": "The job type for the current run."}
    )
