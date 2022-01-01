#!/bin/sh -eu
INPUT_GCS_PATTERN=$1
# e.g., gs://tfrecords/train-*
OUT_GCS_DIR=$2
# e.g., gs://models/model_name

if [ -n "${TPU_CONFIG_JSON+x}" ]; then
    if [ -e "$TPU_CONFIG_JSON" ]; then
        NAME=$(cat "$TPU_CONFIG_JSON" | jq -r '.name')
    else
        echo "\$TPU_CONFIG_JSON doesn't exist; $TPU_CONFIG_JSON"
        exit 1
    fi
else
    echo "\$TPU_CONFIG_JSON is undefined. You must set it like:"
    echo ""
    echo "    export TPU_CONFIG_JSON=tpu_configs/foo.json"
    exit 1
fi

PYTHONPATH=tf_tpu_models \
    python tf_tpu_models/official/detection/main.py \
    --mode="train" \
    --model="mask_rcnn" \
    --model_dir="$OUT_GCS_DIR" \
    --eval_after_training=False \
    --config_file=configs/spinenet/sn143-imat-v2.yaml \
    --use_tpu=True \
    --tpu="$NAME" \
    --params_override="train.train_file_pattern=$INPUT_GCS_PATTERN"
