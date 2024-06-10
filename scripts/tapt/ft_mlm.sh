#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

MODEL=Davlan/afro-xlmr-large-61L

# eng
for LANG in eng arq ary amh hau kin esp tel mar; do
  OUTPUT_DIR=$PROJECT_DIR/checkpoints/afro_ft/mlm/${LANG}

  python $PROJECT_DIR/run_mlm.py \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --train_file $PROJECT_DIR/data/tapt/${LANG}.txt \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --logging_steps 100 \
    --save_strategy no  \
    --line_by_line  \
    --overwrite_output_dir

done