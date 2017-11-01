#!/usr/bin/env bash

conda env create -c conda-forge -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter
