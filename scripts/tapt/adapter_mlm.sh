#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_MODE=disabled

MODEL=Davlan/afro-xlmr-large-61L

for LANG in tel; do
  OUTPUT_DIR=checkpoints/afro_adapter/lang_adapter/tapt/${LANG}

  python $PROJECT_DIR/train_lang_adapter.py \
              --language $LANG  \
              --model_name_or_path $MODEL \
              --tokenizer_name $MODEL  \
              --train_file $PROJECT_DIR/data/tapt/${LANG}.txt  \
              --per_device_train_batch_size 16 \
              --num_train_epochs 10  \
              --do_train  \
              --logging_steps 10 \
              --save_strategy no  \
              --line_by_line  \
              --output_dir $OUTPUT_DIR \
              --overwrite_output_dir \
              --train_adapter \
              --load_adapter checkpoints/afro_adapter/lang_adapter/leipzig/${LANG}/${LANG}
done
