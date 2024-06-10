#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for regression."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
import wandb
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from scipy.stats import spearmanr

import transformers
from adapters import AdapterArguments, AdapterTrainer, AutoAdapterModel, setup_adapter_training, AdapterConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from utils import load_str_dataset, load_stsb_dataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "STSB" : ("sentence1", "sentence2"),
    "STR" : ("sentence1", "sentence2"),
    'translated-STR' : ("sentence1", "sentence2")
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments :
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default='STR',
        metadata={"help" : "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
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


@dataclass
class ModelArguments :
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help" : "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help" : "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help" : "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main() :
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json") :
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else :
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = transformers.logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir :
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0 :
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None :
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: Loading a dataset from your local files.
    data_files = {"train" : data_args.train_file,
                  "validation" : data_args.validation_file}

    if data_args.task_name.lower() == 'stsb' :
        raw_datasets = load_stsb_dataset(data_files={"train": data_args.train_file}, language=data_args.language)
        raw_datasets['validation'] = load_str_dataset(data_files={"validation": data_args.validation_file})
    else :
        raw_datasets = load_str_dataset(data_files=data_files)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name
                                              if model_args.tokenizer_name else model_args.model_name_or_path)
    # We use the AutoAdapterModel class here for better adapter support.
    model = AutoAdapterModel.from_pretrained(model_args.model_name_or_path,
                                             cache_dir=model_args.cache_dir)

    # regression head
    adapter_name = data_args.language + '-' + data_args.task_name
    if training_args.do_train:
        model.add_classification_head(
            adapter_name,
            activation_function='sigmoid',
            num_labels=1,
            id2label=None
        )
    else:
        assert len(adapter_args.load_adapter)>0
        model.load_head(adapter_args.load_adapter)   # we saved it in the same path as the task adapter

    # Setup adapters
    setup_adapter_training(model, adapter_args, adapter_name)

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    if data_args.max_seq_length > tokenizer.model_max_length :
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples) :
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
        return result

    with training_args.main_process_first(desc="dataset map pre-processing") :
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train :
        if "train" not in raw_datasets :
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval :
        if "validation" not in raw_datasets :
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    def compute_metrics(p: EvalPrediction) :
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        mse = ((preds - p.label_ids) ** 2).mean().item()
        spearman = spearmanr(preds, p.label_ids)[0]
        return {"mse" : mse, "spearmanr" : spearman}

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train :
        checkpoint = None
        if training_args.resume_from_checkpoint is not None :
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None :
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        wandb.log({"eval/best_metric" : trainer.state.best_metric})

    # Evaluation
    if training_args.do_eval :
        logger.info("*** Evaluate ***")
        # the best checkpoint should be loaded because "load_best_at_end=True"
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        wandb.log({"eval/best_score" : metrics['eval_spearmanr']})  # should be equal to self.state.best_metric


def _mp_fn(index) :
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__" :
    main()
