#!/bin/sh -eu
export STORAGE_BUCKET=gs://strf-sandbox/hiroshi.matsui/segmentation/kaggle-imaterialist-2020
export MODEL_DIR="$STORAGE_BUCKET"/models

PYTHONPATH=tf_tpu_models \
    python tf_tpu_models/official/detection/main.py \
    --model="mask_rcnn" \
    --model_dir="$MODEL_DIR" \
    --mode="train" \
    --eval_after_training=False \
    --config_file=configs/spinenet/sn143-imat-v2.yaml \
    --use_tpu=True \
    --tpu="kaggle-imat2020"
