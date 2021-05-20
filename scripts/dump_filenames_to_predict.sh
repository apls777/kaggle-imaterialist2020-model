#!/bin/sh -eu
IMAGES_DIR=$1
ls "$IMAGES_DIR" \
    | sort \
    | awk 'BEGIN {print "image_id,filename"} {print NR-1","$1}'
    # | awk '{print "{\"image_id\": "NR-1", \"filename\": \""$1"\"}"}' \
