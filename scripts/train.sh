#!/bin/sh -eu
INPUT_GCS_PATTERN=$1
# e.g., gs://tfrecords/train-*
OUT_GCS_DIR=$2
# e.g., gs://models/model_name

PYTHONPATH=tf_tpu_models \
    python tf_tpu_models/official/detection/main.py \
    --mode="train" \
    --model="mask_rcnn" \
    --model_dir="$OUT_GCS_DIR" \
    --eval_after_training=False \
    --config_file=configs/spinenet/sn143-imat-v2.yaml \
    --use_tpu=True \
    --tpu="kaggle-imat2020" \
    --params_override="train.train_file_pattern=$INPUT_GCS_PATTERN"
