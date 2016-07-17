#!/usr/bin/env bash
conda config --add channels conda-forge
conda env create -n sklearn.tree.model.export python=2 -f environment.yml
source activate sklearn.tree.model.export