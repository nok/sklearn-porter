#!/usr/bin/env bash

if ! conda env list | grep sklearn-porter ; then
    SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
    conda env create \
        -n sklearn-porter \
        -c defaults \
        python=3.5  # python 3.5 is the lowest supported python version
    conda run -n sklearn-porter \
        pip install --upgrade pip
fi
