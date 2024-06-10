#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_PROJECT=semeval
export WANDB_RUN_GROUP=ft_stsb

MODEL=Davlan/afro-xlmr-large-61L
OUTPUT=${PROJECT_DIR}/checkpoints/afro_ft/stsb

for LANG in arq kin eng amh ary tel hau esp mar; do
  export WANDB_JOB_TYPE=${LANG}

  for LR in 2e-5 5e-5; do
    export WANDB_NAME=${LR}
    OUTPUT_DIR=${OUTPUT}/${LANG}_${LR}

    python $PROJECT_DIR/train_ft.py \
            --model_name_or_path $MODEL \
            --language $LANG \
            --train_file $PROJECT_DIR/data/track_a/$LANG/${LANG}_train.csv \
            --validation_file $PROJECT_DIR/data/track_a/$LANG/${LANG}_dev_with_labels.csv \
            --augmentation_file $PROJECT_DIR/data/stsb/${LANG}_nllb/train.csv  \
            --per_device_train_batch_size 16 \
            --learning_rate $LR \
            --num_train_epochs 6 \
            --eval_steps 50 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir
  done
done
