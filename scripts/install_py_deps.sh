#!/bin/bash -ex
SCRIPTS_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

pip install -U pip

pip install poetry

poetry install

"$SCRIPTS_DIR"/install_tfrecord_requirements.sh
