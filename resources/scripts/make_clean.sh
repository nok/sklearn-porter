#!/usr/bin/env bash

source "$(cd "$(dirname "$0")"; pwd -P)"/source_me.sh

find . -name '.pytest_cache' -type d -delete
find . -name '__pycache__' -type d -delete
find . -name '*.pyc' -type f -delete
find . -name '*.DS_Store' -type f -delete
find . -name 'tmp' -maxdepth 1 -type d -delete
rm -rf build dist sklearn_porter.egg-info
