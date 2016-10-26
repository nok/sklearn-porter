#!/usr/bin/env bash
source activate sklearn-porter
python -m unittest discover -p '*Test.py'
source deactivate