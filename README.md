# Kaggle iMaterialist 2020

This repository is forked from https://github.com/apls777/kaggle-imaterialist2020-model for me to easily use it.

## Setup

Create a VM instance and TPU by the following:

```sh
./scripts/tpu.sh create $GCP_PROJECT
```

Grant `roles/storage.admin` the TPU service account (e.g., `service-1123456789@cloud-tpu.iam.gserviceaccount.com`).

You can know the other sub-commands for the VM and TPU by the following:

```sh
./scripts/tpu.sh -h
```

Install Python dependencies.

```
poetry install
poetry shell
```

Install `ctpu` command accroding to the following instructions.

https://github.com/tensorflow/tpu/tree/master/tools/ctpu#download

Create TF Records from *iMaterialist* dataset with [*COCO* format annotations](https://github.com/cvdfoundation/fashionpedia#annotations) by the following scripts.

```sh
# install dependencies.
./scripts/install_tfrecord_requirements.sh

# create TF Records from the images and the annotation JSON file.
./scripts/create_tf_records.sh $IMAGE_DIR $COCO_JSON_FILE $OUTPUT_FILE_PREFIX

# e.g., You can pass the arguments like the following if you use https://github.com/hrsma2i/dataset-iMaterialist
DATASET_ROOT=dataset-iMaterialist/raw
./scripts/create_tf_records.sh \
    $DATASET_ROOT/train \
    $DATASET_ROOT/instances_attributes_train2020.json \
    $DATASET_ROOT/tfrecords/train
```

The above script creates 50 shards of TF Record like `$DATASET_ROOT/tfrecords/image-annotation-00001-of-00050.tfrecord`.
