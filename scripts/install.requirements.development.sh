#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

pip freeze | grep --quiet twine
if [[ $? -eq 1 ]]; then
    pip install --no-cache-dir \
        -r ${SCRIPT_PATH}/../requirements.development.txt
fi
