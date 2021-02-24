#!/usr/bin/env bash

source "$(cd "$(dirname "$0")"; pwd -P)"/source_me.sh

pytest tests -v \
  --cov=sklearn_porter \
  --disable-warnings \
  --numprocesses=auto \
  -p no:doctest \
  -o python_files="EstimatorTest.py" \
  -o python_functions="test_*"

exit $?
