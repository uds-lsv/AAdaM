"""Fine-tuning the full model ( cross-encoder) with provided dataset."""

import logging
import os
import math

from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from sentence_transformers import (
    LoggingHandler
)

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

from utils import create_str_dataset, create_stsb_dataset, train_callback, eval_callback
from arguments import ModelArguments, DataTrainingArguments, SemArguments, WandbArguments

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

from dataclasses import dataclass, field
from typing import Optional

import wandb

@dataclass
class DataTrainingArguments :
    language: Optional[str] = field(
        default='eng',
        metadata={"help": "The language of the task"}
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help" : (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help" : "Overwrite the cached preprocessed datasets or not."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help" : "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help" : "A csv or a json file containing the validation data."}
    )
    augmentation_file: Optional[str] = field(
        default=None, metadata={"help" : "A csv or a json file containing the validation data."}
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


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.info(model_args)
    logging.info(data_args)

    # create output dir
    os.makedirs(training_args.output_dir, exist_ok=True)

    # set seed before initializing model.
    set_seed(training_args.seed)

    # load data
    train_samples = create_str_dataset(data_args.train_file)
    eval_samples = create_str_dataset(data_args.validation_file)
    logging.info("Train samples: {}".format(len(train_samples)))

    aug_samples = None
    if data_args.augmentation_file is not None:
        if 'stsb' in data_args.augmentation_file:
            aug_samples = create_stsb_dataset(data_args.augmentation_file, data_args.language)
        elif 'translate' in data_args.augmentation_file:
            aug_samples = create_str_dataset(data_args.augmentation_file)
        else:
            raise NotImplementedError

    model = CrossEncoder(model_args.model_name_or_path, num_labels=1, max_length=data_args.max_seq_length)
    evaluator = CECorrelationEvaluator.from_input_examples(eval_samples, name='dev')

    # training
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=training_args.per_device_train_batch_size)
    warmup_steps = math.ceil(
        len(train_dataloader) * training_args.num_train_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    wandb.init(project=os.environ["WANDB_PROJECT"],
               name=os.environ["WANDB_NAME"],
               job_type=os.environ["WANDB_JOB_TYPE"],
               group=os.environ["WANDB_RUN_GROUP"])

    if aug_samples is not None:
        aug_dataloader = DataLoader(aug_samples, shuffle=True, batch_size=training_args.per_device_train_batch_size)
        model.fit(train_dataloader=aug_dataloader,
                  evaluator=evaluator,
                  epochs=int(training_args.num_train_epochs),
                  optimizer_params={'lr' : training_args.learning_rate},
                  evaluation_steps=training_args.eval_steps,
                  warmup_steps=int(warmup_steps),
                  output_path=training_args.output_dir,
                  save_best_model=True)
        del model
        model = CrossEncoder(training_args.output_dir, num_labels=1)

    model.fit(train_dataloader=train_dataloader,
                  evaluator=evaluator,
                  epochs=int(training_args.num_train_epochs),
                  optimizer_params={'lr': training_args.learning_rate},
                  evaluation_steps=training_args.eval_steps,
                  warmup_steps=warmup_steps,
                  output_path=training_args.output_dir,
                  save_best_model=True,
                  log_callback=train_callback,
                  callback=eval_callback)

if __name__ == '__main__':
    main()
