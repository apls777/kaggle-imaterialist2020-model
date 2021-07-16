#!/bin/sh -eux
MODEL_GCS_DIR=$1
# e.g., gs://models/sn143-imat-v2
INPUT_GCS_PATTERN=$2
# e.g., gs://tfrecords/predict-*
OUTPUT_DIR=$3
# e.g., gs://predictions

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
    --mode="predict" \
    --model="mask_rcnn" \
    --model_dir="$MODEL_GCS_DIR" \
    --predict_checkpoint_step=200000 \
    --predict_file_pattern="$INPUT_GCS_PATTERN" \
    --predict_output_dir="$OUTPUT_DIR" \
    --config_file="$MODEL_GCS_DIR"/params.yaml \
    --use_tpu=True \
    --tpu="$NAME" \
    --params_override="eval.use_json_file=False, eval.val_json_file=''"
