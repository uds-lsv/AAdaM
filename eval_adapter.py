import os
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional

from adapters import AdapterTrainer, AutoAdapterModel
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

from adapters import Stack

from utils import load_str_dataset

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
    track: Optional[str] = field(
        default='c',
        metadata={"help" : "The track of the task, a or c"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help" : "A csv or a json file containing the validation data."}
    )


@dataclass
class ModelArguments :
    model_name_or_path: str = field(
        default='Davlan/afro-xlmr-large-61L',
        metadata={"help" : "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help" : "Pretrained tokenizer name or path if not the same as model_name"}
    )
    task_adapter_dir: Optional[str] = field(
        default=None, metadata={"help" : "The directory to load the task adapter from."}
    )
    task_adapter_list: Optional[str] = field(
        default=None, metadata={"help" : "The task adapter(s) to evaluate separated by comma. We will ensemble their results."}
    )
    lang_adapter: Optional[str] = field(
        default=None, metadata={"help" : "The language adapter to use."}
    )


def compute_metrics(p: EvalPrediction) :
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds)
    mse = ((preds - p.label_ids) ** 2).mean().item()
    spearman = spearmanr(preds, p.label_ids)[0]
    return {"mse" : mse, "spearmanr" : spearman}

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(data_args)

    os.makedirs(training_args.output_dir, exist_ok=True)

    ads = [ad.strip() for ad in model_args.task_adapter_list.split(',')]
    task_adapter_list = [os.path.join(model_args.task_adapter_dir, ad) for ad in ads if len(ad) > 0]
    print(task_adapter_list)

    output_file = os.path.join(training_args.output_dir, f"pred_{data_args.language}_{data_args.track}.csv")

    # get dataset
    data_files = {"test" : data_args.validation_file}
    test_dataset = load_str_dataset(data_files=data_files)['test']
    sentence1_key, sentence2_key = 'sentence1', 'sentence2'
    labels = test_dataset['label']

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoAdapterModel.from_pretrained(model_args.model_name_or_path)
    lang_adapter_name = model.load_adapter(model_args.lang_adapter, with_head=False)

    # process dataset
    def preprocess_function(examples) :
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length", max_length=256, truncation=True)
        return result

    test_dataset = test_dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset")

    # evaluate
    metrics = []
    pred_scores = []
    for task_adapter in task_adapter_list :
        print(f"Evaluating {task_adapter} ... ")
        task_adapter_name = model.load_adapter(task_adapter, with_head=False)
        model.load_head(task_adapter)
        model.set_active_adapters(Stack(lang_adapter_name, task_adapter_name))
        #print("Active adapters: ", model.active_adapters)

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions)
        pred_scores.append(predictions)

        if training_args.do_eval: # only support labeled datasets
            metric = spearmanr(predictions, labels)[0] * 100
            metrics.append(metric)

        model.delete_adapter(task_adapter)
        model.delete_head(task_adapter)

    ensemble_scores = np.mean(np.array(pred_scores), axis=0)

    data = {
        'PairID' : test_dataset['idx'],
        'Pred_Score' : ensemble_scores
    }
    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_file, index=False)

    log_scores = np.transpose(np.array(pred_scores))
    log_scores = np.array([str(l).strip("[]") for l in log_scores])
    log_data = {
        'PairID': test_dataset['idx'],
        'Pred_Score': ensemble_scores,
        'Log_Scores': log_scores
    }
    log_df = pd.DataFrame.from_dict(log_data)
    log_df.to_csv(output_file+'.log', index=False)

    if training_args.do_eval:
        print(f"spearmanr: {metrics}")
        print(f"Avg/std spearmanr: {round(np.mean(metrics), 2)} ± {round(np.std(metrics), 2)}")
        with open(output_file + '.metric', 'a') as f :
            f.write(f"Task adapter: {model_args.task_adapter_list}\n")
            f.write(f"spearmanr: {metrics}\n")
            f.write(f"Avg/std spearmanr: {round(np.mean(metrics), 2)} ± {round(np.std(metrics), 2)}\n")


if __name__ == "__main__" :
    main()
