#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

jupyter notebook \
  --notebook-dir='examples/basics' \
  --ip='0.0.0.0' \
  --port=8888
