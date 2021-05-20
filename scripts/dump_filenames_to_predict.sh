#!/bin/sh -eux
IMAGES_DIR=$1
PREDICTION_GCS_DIR=$2
ls "$IMAGES_DIR" \
    | sort \
    | awk 'BEGIN {print "image_id,filename"} {print NR-1","$1}' \
    > /tmp/image_id_filename.csv

gsutil -m cp /tmp/image_id_filename.csv "$PREDICTION_GCS_DIR"
