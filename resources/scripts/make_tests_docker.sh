#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

docker build \
  -t sklearn-porter:1.0.0 \
  --build-arg PYTHON_VER=${PYTHON_VER:-python=3.6} \
  --build-arg CYTHON_VER=${CYTHON_VER:-cython} \
  --build-arg NUMPY_VER=${NUMPY_VER:-numpy} \
  --build-arg SCIPY_VER=${SCIPY_VER:-scipy} \
  --build-arg SKLEARN_VER=${SKLEARN_VER:-scikit-learn==0.21} \
  .

docker run \
  -v $(pwd):/home/abc/repo \
  --detach \
  --entrypoint=/bin/bash \
  --name test \
  -t sklearn-porter

docker exec -it test ./docker-entrypoint.sh \
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
