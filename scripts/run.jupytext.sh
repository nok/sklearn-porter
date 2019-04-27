#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/../examples/estimator

for py_file in $(find . -type f -name '*.pct.py')
do
    ipynp_file="${py_file%.py}.ipynb"
    echo "$py_file"
    echo "$ipynp_file"
    jupytext \
        --from "py:percent" \
        --to "notebook" "$py_file"
    jupyter nbconvert \
        --to notebook \
        --execute "$ipynp_file" \
        --output $(basename -- "$ipynp_file")
done

for json_file in $(find . -type f -name 'data.json')
do
    echo "$json_file"
    rm -f "$json_file"
done
