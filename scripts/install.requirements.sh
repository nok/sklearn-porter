#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pip freeze | grep --quiet scikit-learn
if [[ $? -eq 1 ]]; then
    pip install --no-cache-dir \
        -r $SCRIPTPATH/../requirements.txt
fi
