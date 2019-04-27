#!/usr/bin/env bash

if ! conda env list | grep sklearn-porter ; then
    SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
    conda env create \
        -n sklearn-porter \
        -f $SCRIPTPATH/../environment.yml
    conda run -n sklearn-porter \
        pip install --upgrade pip
fi
