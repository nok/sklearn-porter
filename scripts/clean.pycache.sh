#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd ${SCRIPT_PATH}/..

find . -name '__pycache__' -type d -delete
find . -name '*.pyc' -type f -delete
