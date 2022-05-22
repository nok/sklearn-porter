#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

pytest tests -v \
  --cov=sklearn_porter \
  --disable-warnings \
  --numprocesses=auto \
  -p no:doctest \
  -o python_files="EstimatorTest.py" \
  -o python_functions="test_*"

exit $?
