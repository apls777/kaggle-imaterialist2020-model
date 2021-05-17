#!/bin/sh -eux
MODE=$1
IMAGE_DIR=$2
COCO_JSON_FILE=$3
OUTPUT_FILE_PREFIX=$4
if [ "$MODE" = "train" ]; then
    PYTHONPATH="tf-models:tf-models/research" \
        python tools/datasets/create_coco_tf_record.py \
        --logtostderr \
        --include_masks \
        --image_dir="$IMAGE_DIR" \
        --object_annotations_file="$COCO_JSON_FILE" \
        --output_file_prefix="$OUTPUT_FILE_PREFIX" \
        --num_shards=50
elif [ "$MODE" = "predict" ]; then
    PYTHONPATH="tf-models:tf-models/research" \
        python tools/datasets/create_coco_tf_record.py \
        --logtostderr \
        --image_dir="$IMAGE_DIR" \
        --output_file_prefix="$OUTPUT_FILE_PREFIX" \
        --num_shards=50
else
    echo "invalid mode: $MODE"
fi
