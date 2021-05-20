# Kaggle iMaterialist 2020

This repository is forked from https://github.com/apls777/kaggle-imaterialist2020-model for me to easily use it.

<!-- TOC -->

- [Kaggle iMaterialist 2020](#kaggle-imaterialist-2020)
    - [Setup](#setup)
- [Train](#train)
- [Predict](#predict)

<!-- /TOC -->

## Setup

Install `ctpu` command.

https://github.com/tensorflow/tpu/tree/master/tools/ctpu#download


Create a VM instance and TPU.

```sh
./scripts/tpu.sh create $GCP_PROJECT
```

Grant `roles/storage.admin` the TPU service account (e.g., `service-1123456789@cloud-tpu.iam.gserviceaccount.com`).

Log in to the instance via SSH.

```
./scripts/tpu.sh ssh
```

You can know the other sub-commands for the VM and TPU by the following:

```sh
./scripts/tpu.sh -h
```

Install Python dependencies to the instance.

```
pip install poetry
poetry install
poetry shell
```

# Train

Set up [iMaterialist](https://github.com/hrsma2i/dataset-iMaterialist) dataset.

Install other dependencies.

```sh
./scripts/install_tfrecord_requirements.sh
```

Create TF Records from iMaterialist [*COCO* format annotations](https://github.com/cvdfoundation/fashionpedia#annotations).

```sh
PYTHONPATH="tf-models:tf-models/research" \
    python tools/datasets/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --num_shards=50 \
    --image_dir=$IMAGE_DIR \
    --object_annotations_file=$COCO_JSON_FILE \
    --output_file_prefix=$OUTPUT_FILE_PREFIX

# e.g.,
PYTHONPATH="tf-models:tf-models/research" \
    python tools/datasets/create_coco_tf_record.py \
    --logtostderr \
    --include_masks \
    --num_shards=50 \
    --image_dir=~/iMaterialist/raw/train \
    --object_annotations_file=~/iMaterialist/raw/instances_attributes_train2020.json \
    --output_file_prefix=gs://yourbucket/tfrecords/train
```

TF Records will be created like `gs://yourbucket/tfrecords/train-00001-of-00050.tfrecord`.

Train a model.

```sh
./scripts/train.sh $INPUT_GCS_PATTERN $OUT_GCS_DIR

# e.g.,
./scripts/train.sh \
    gs://yourbucket/tfrecords/train-* \
    gs://yourbucket/model
```

Training artifacts (checkpoints, hyperparmeters, and logs etc) will be dumped into `gs://yourbucket/model/`.

Don't forget to **stop your TPU** if the training finishes.

```sh
./scripts/tpu.sh stop --tpu
```

**Warning**

If the training fails, delete the training artifacts from GCS. Otherwise, the configurations of the failed training will be loaded and it will fail again. For example, tensor's shape mismatch.

# Predict

Predict for your images.

```sh
./scripts/predict.sh \
    $MODEL_GCS_DIR \
    $IMAGE_GCS_DIR \
    $TF_RECORD_GCS_DIR \
    $OUT_GCS_DIR

# e.g.,
./scripts/predict.sh \
    gs://yourbucket/model \
    gs://yourbucket/yourdataset/images \
    gs://yourbucket/yourdataset/tfrecords \
    gs://yourbucket/yourdataset/predictions
```

The TPU should be automatically shut down by `scripts/predict.sh`.

The prediction results are dumped to `gs://yourbucket/yourdataset/predictions/predictions.json` (JSON Lines).
Its schema is:

```json
{
  "image_id": 0,
  "category_id": 32,
  "bbox": [
    382.5208129883,
    660.4463500977,
    156.1093292236,
    122.5846252441
  ],
  "score": 0.984375,
  "segmentation": {
    "size": [
      1024, // height
      839   // width
    ],
    "counts": "<compressed RLE>"
  },
  "mask_mean_score": 0.9516192675,
  "mask_area_fraction": 0.7079081535,
  "attribute_probabilities": [
    0.0067976713,
    0.0062188804,
    //...
    0.0126225352
  ],
  "id": 1,
  "filename": "example.jpg"
}
{
    //...
}
//...
```

You can get the binary mask `numpy.ndarray` for a particular category using [hrsma2i/segmentation](https://github.com/hrsma2i/segmentation#coco-rle--numpyndarray).
