#!/bin/bash -ex
{
    echo 'export EDITOR=vim'
    echo 'export PATH=$HOME/.local/bin:$PATH'
    echo 'alias python=python3'
    echo 'alias pip=pip3'
} >>~/.bashrc
