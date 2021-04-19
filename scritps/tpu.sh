#!/bin/sh -eu

PROJECT_ID=$1

# Tensorflow version may be 1.13
# c.f., https://github.com/apls777/kaggle-imaterialist2020-model/blob/7466434d719b346b04ea0cde8c121d45c1338ce1/tf_tpu_models/official/mask_rcnn/mask_rcnn_k8s.yaml#L25:w
ctpu up --project="$PROJECT_ID" \
    --zone=us-central1-a \
    --disk-size-gb=300 \
    --machine-type=n1-standard-8 \
    --name=kaggle-imat2020 \
    --tf-version=1.15.5 \
    --tpu-size=v3-8
