#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd ${SCRIPT_PATH}/..

pytest tests -v \
    --cov=sklearn_porter \
    --disable-warnings \
    --numprocesses=auto \
    -p no:doctest \
    -o python_files="EstimatorTest.py" \
    -o python_functions="test_*"

exit $?
