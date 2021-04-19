#!/bin/sh -eu

PROJECT_ID=$1

ctpu up --project="$PROJECT_ID" \
    --zone=us-central1-a \
    --vm-only \
    --disk-size-gb=300 \
    --machine-type=n1-standard-8 \
    --name=kaggle-imat2020 \
    --tf-version=2.4.1
