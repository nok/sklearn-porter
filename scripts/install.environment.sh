#!/usr/bin/env bash

if ! conda env list | grep sklearn-porter ; then
    SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
    conda env create \
        -n sklearn-porter \
        -f ${SCRIPT_PATH}/../environment.yml
    conda run -n sklearn-porter \
        pip install --upgrade pip
fi
