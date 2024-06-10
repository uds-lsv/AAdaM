#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_PROJECT=semeval
export WANDB_RUN_GROUP=lang_adapter
export WANDB_JOB_TYPE=leipzig
export WANDB_NOTES="pre-train a language adapter using the Leipzig corpora"

MODEL=Davlan/afro-xlmr-large-61L

for LANG in arb; do
  export WANDB_NAME=$LANG

  TRAIN_FILE=data/monolingual/${LANG}.txt
  OUTPUT_DIR=checkpoints//afro_adapter/lang_adapter/leipzig/${WANDB_NAME}

  python $PROJECT_DIR/train_lang_adapter.py \
              --language $LANG  \
              --model_name_or_path $MODEL \
              --tokenizer_name $MODEL  \
              --train_file $TRAIN_FILE  \
              --per_device_train_batch_size 16 \
              --gradient_accumulation_steps 4  \
              --max_steps 50000  \
              --do_train  \
              --logging_steps 100 \
              --save_strategy no  \
              --line_by_line  \
              --output_dir $OUTPUT_DIR \
              --overwrite_output_dir \
              --train_adapter \
              --adapter_config seq_bn_inv
done