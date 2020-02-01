#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd ${SCRIPT_PATH}/..

# Directories:
find . -iname '.pytest_cache' -type d -exec rm -rf "{}" +
find . -name '__pycache__' -type d -exec rm -rf "{}" +

# Files:
find . -name '*.pyc' -type f -exec rm -rf "{}" \;
find . -iname '*.DS_Store' -type f -exec rm -rf "{}" \;
