#!/bin/sh -eux
IMAGE_DIR=$1
COCO_JSON_FILE=$2
OUT_TF_RECORD_FILE=$3
PYTHONPATH="tf-models:tf-models/research" \
    python tools/datasets/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --image_dir="$IMAGE_DIR" \
    --object_annotations_file="$COCO_JSON_FILE" \
    --output_file_prefix="$OUT_TF_RECORD_FILE" \
    --num_shards=50
