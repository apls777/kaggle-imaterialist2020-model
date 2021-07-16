#!/bin/bash -ex
SCRIPTS_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

{
    echo 'export EDITOR=vim'
    echo 'export PATH=$HOME/.local/bin:$PATH'
    echo 'alias python=python3'
    echo 'alias pip=pip3'
} >>~/.bashrc

source ~/.bashrc

pip install -U pip

pip install poetry

poetry install

$SCRIPTS_DIR/install_tfrecord_requirements.sh
