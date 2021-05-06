#!/bin/sh -eux
IMAGE_DIR=$1
COCO_JSON_FILE=$2
OUTPUT_FILE_PREFIX=$3
PYTHONPATH="tf-models:tf-models/research" \
    python tools/datasets/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --image_dir="$IMAGE_DIR" \
    --object_annotations_file="$COCO_JSON_FILE" \
    --output_file_prefix="$OUTPUT_FILE_PREFIX" \
    --num_shards=50
