#!/usr/bin/env bash

source activate sklearn-porter
python -m unittest discover -vp '*Test.py'
source deactivate