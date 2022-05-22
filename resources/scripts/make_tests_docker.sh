#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

docker build \
    -t sklearn-porter:testing \
    --build-arg PYTHON_VER="${PYTHON_VER:-python=3.6}" \
    --build-arg SKLEARN_VER="${SKLEARN_VER:-scikit-learn}" \
    .

docker run \
    --name "sklearn-porter" \
    --volume $(pwd):/home/user/repo \
    --entrypoint=/bin/bash \
    --detach \
    -t sklearn-porter:testing

docker exec -it "sklearn-porter" ./docker-entrypoint.sh \
  pytest tests -v \
    --cov=sklearn_porter \
    --disable-warnings \
    --numprocesses=auto \
    -p no:doctest \
    -o python_files="EstimatorTest.py" \
    -o python_functions="test_*"

success=$?

docker rm -f $(docker ps --all --filter name="sklearn-porter" -q)

exit ${success}
