#!/bin/bash -eux
sudo apt update && sudo apt install -y protobuf-compiler python3-pil python3-lxml python3-pip python3-dev git unzip jq

SCRIPTS_DIR=$(
    cd "$(dirname "$0")"
    pwd
)
PRJ_ROOT=$(dirname "$SCRIPTS_DIR")

pip install -U pip

pip install poetry

poetry install

pip install Cython
pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI

TF_MODELS_DIR="$PRJ_ROOT"/tf-models
if [ ! -e "$TF_MODELS_DIR" ]; then
    git clone https://github.com/tensorflow/models.git "$TF_MODELS_DIR"
    touch "$TF_MODELS_DIR"/__init__.py
    touch "$TF_MODELS_DIR"/research/__init__.py
fi

(cd "$TF_MODELS_DIR"/research && protoc object_detection/protos/*.proto --python_out=.)
