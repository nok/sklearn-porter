#!/usr/bin/env bash

conda config --add channels conda-forge
conda env create -n sklearn-porter python=2 -f environment.yml
source activate sklearn-porter