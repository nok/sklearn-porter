#!/usr/bin/env bash

source activate sklearn-porter
find ./sklearn_porter -name '*.py' -exec pylint {} \;
source deactivate
