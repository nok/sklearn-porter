#!/usr/bin/env bash

source "$(cd "$(dirname "$0")"; pwd -P)"/source_me.sh

jupyter notebook \
  --notebook-dir='examples/basics' \
  --ip='0.0.0.0' \
  --port=8888
