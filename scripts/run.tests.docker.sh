#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd ${SCRIPT_PATH}/..

docker build \
    -t sklearn-porter \
    --build-arg PYTHON_VER=${PYTHON_VER:-python=3.5} \
    --build-arg CYTHON_VER=${CYTHON_VER:-cython} \
    --build-arg NUMPY_VER=${NUMPY_VER:-numpy} \
    --build-arg SCIPY_VER=${SCIPY_VER:-scipy} \
    --build-arg SKLEARN_VER=${SKLEARN_VER:-scikit-learn} \
    .

docker run \
    -t sklearn-porter \
    --volume $(pwd):/home/abc/repo \
    --name test \
        pytest tests -v \
            --cov=sklearn_porter \
            --disable-warnings \
            --numprocesses=auto \
            -p no:doctest \
            -o python_files="EstimatorTest.py" \
            -o python_functions="test_*"

success=$?

docker rm -f $(docker ps --all --filter name=test -q)

exit ${success}
