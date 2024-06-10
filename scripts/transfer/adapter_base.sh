#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=results/track_c/base_testset_translate_test/
LANG_ADAPTER_DIR=${PROJECT_DIR}/checkpoints/afro_adapter/lang_adapter/leipzig

for TARGET in amh ary hau kin arq hin arb ind afr pan; do
  for SOURCE in eng; do
    VAL_FILE=${PROJECT_DIR}/data/translate_test/${SOURCE}/${TARGET}/${TARGET}_test_with_labels.csv

    TASK_ADAPTER_DIR=${PROJECT_DIR}/checkpoints/track_c/ad_base
    TASK_ADAPTERS=${SOURCE}/${SOURCE}-STR
    LANG_ADAPTER=${LANG_ADAPTER_DIR}/${TARGET}/${TARGET}

    python $PROJECT_DIR/eval_adapter.py \
        --language $TARGET \
        --track c \
        --validation_file $VAL_FILE  \
        --task_adapter_dir $TASK_ADAPTER_DIR \
        --task_adapter_list $TASK_ADAPTERS \
        --lang_adapter $LANG_ADAPTER \
        --per_device_eval_batch_size 200 \
        --output_dir $OUTPUT_DIR \
        --do_eval
  done
done

