#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

pip freeze | grep --quiet twine
if [[ $? -eq 1 ]]; then
    cd ${SCRIPT_PATH}/..
    pip install --no-cache-dir -e .[development]
fi
