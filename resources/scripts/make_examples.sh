#!/usr/bin/env bash

source "$(cd "$(dirname "$0")"; pwd -P)"/source_me.sh

cd examples

for py_file in $(find . -type f -name '*.pct.py')
do
  ipynp_file="${py_file%.py}.ipynb"
  echo "$py_file -> $ipynp_file"

  jupytext \
    --from "py:percent" \
    --to "notebook" "$py_file"

  jupyter nbconvert \
    --to notebook \
    --config examples/.ipython_config.py \
    --execute "$ipynp_file" \
    --output $(basename -- "$ipynp_file")
done
