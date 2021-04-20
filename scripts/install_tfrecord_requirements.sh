#!/bin/sh -eu
sudo apt update && sudo apt install -y protobuf-compiler python3-pil python3-lxml python3-pip python3-dev git unzip

pip install Cython
pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI

if [ ! -e tf-models ]; then
    git clone https://github.com/tensorflow/models.git tf-models
    touch tf-models/__init__.py
    touch tf-models/research/__init__.py
fi

(cd tf-models/research && protoc object_detection/protos/*.proto --python_out=.)
