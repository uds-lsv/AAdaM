#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/SemEval
source ${PROJECT_DIR}/scripts/setup.sh

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

for TARGET in eng; do
  for SOURCE in amh mar ary tel hau esp kin arq; do
      VAL_FILE=${PROJECT_DIR}/data/track_a/${TARGET}/${TARGET}_dev_with_labels.csv
      MODEL_DIR=${PROJECT_DIR}/checkpoints/best_ft_noaug/
      MODELS=${MODEL_DIR}/${SOURCE}

      OUTPUT_DIR=results/best_ft_noaug/track_c/

      python $PROJECT_DIR/eval_model.py \
          --model_list $MODELS \
          --track c \
          --eval_lang TARGET \
          --data_file $VAL_FILE \
          --output_dir $OUTPUT_DIR \
          --compute_metrics
  done
done