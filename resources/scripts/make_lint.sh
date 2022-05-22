#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

FILES=$(find ./sklearn_porter -name '*.py' -type f | tr '\n' ' ')

pylint \
  --rcfile=.pylintrc \
  --output-format=text ${FILES} 2>&1 \
    | tee pylint.txt | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p'

exit $?
