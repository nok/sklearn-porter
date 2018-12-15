#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pip freeze | grep --quiet twine
if [[ $? -eq 1 ]]; then
    pip install -q --no-cache-dir -r $SCRIPTPATH/../requirements.development.txt
fi
