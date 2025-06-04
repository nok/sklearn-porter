#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

conda env create -n sklearn-porter -f $SCRIPTPATH/../environment.yml
source activate sklearn-porter
