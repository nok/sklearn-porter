#!/usr/bin/env bash

# Create new conda environment:
conda env create -n sklearn-porter -c conda-forge python=2 -f environment.yml

# Or:
# conda create -n sklearn-porter -c conda-forge python=2 scikit-learn pylint jupyter nb_conda twine

# Activate the new environment:
source activate sklearn-porter
