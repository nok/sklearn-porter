#!/usr/bin/env bash

# Activate the relevant environment:
source activate sklearn-porter

# Run linting:
find ./sklearn_porter -name '*.py' -exec pylint {} \;

# Deactivate the previous activated environment:
source deactivate
