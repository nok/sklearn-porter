#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd ${SCRIPT_PATH}/..

find . -name 'tmp' -type d -delete
rm -rf build
rm -rf dist
rm -rf sklearn_porter.egg-info
