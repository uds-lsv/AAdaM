#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_PROJECT=semeval
export WANDB_RUN_GROUP=adapter_trans

MODEL=Davlan/afro-xlmr-large-61L

OUTPUT=${PROJECT_DIR}/checkpoints/afro_adapter/task_adapter/trans
# start from the task adaptive pretrained language adapter
LANG_ADAPTER_DIR=${PROJECT_DIR}/checkpoints/afro_adapter/lang_adapter/leipzig

for LANG in arq kin eng amh mar ary tel hau esp; do
  export WANDB_JOB_TYPE=${LANG}

 for LR in 1e-4 2e-4 5e-5; do
    export WANDB_NAME=${LR}
    OUTPUT_DIR=${OUTPUT}/${LANG}_${LR}

    TRAIN_FILE=data/track_a/$LANG/${LANG}_train.csv
    VAL_FILE=data/track_a/$LANG/${LANG}_dev_with_labels.csv
    AUG_FILE=data/translate_train/${LANG}_nllb/${LANG}_train.csv

    python $PROJECT_DIR/train_warmup_adapter.py \
              --language $LANG  \
              --task_name STR \
              --model_name_or_path $MODEL \
              --tokenizer_name $MODEL  \
              --cache_dir $ADAPTER_CACHE \
              --train_file $TRAIN_FILE  \
              --validation_file $VAL_FILE  \
              --augmentation_file $AUG_FILE  \
              --per_device_train_batch_size 16 \
              --per_device_eval_batch_size  16 \
              --learning_rate $LR \
              --num_train_epochs 15 \
              --do_train  \
              --do_eval  \
              --logging_steps 10 \
              --evaluation_strategy epoch \
              --save_strategy epoch \
              --load_best_model_at_end  \
              --metric_for_best_model spearmanr \
              --save_total_limit  1 \
              --output_dir $OUTPUT_DIR \
              --overwrite_output_dir \
              --train_adapter \
              --adapter_config "seq_bn[reduction_factor=8]" \
              --load_lang_adapter ${LANG_ADAPTER_DIR}/$LANG/$LANG
 done
done
