#!/usr/bin/env bash

source "$(dirname "$(realpath "$0")")"/source_me.sh

if [ -z "$1" ]
then
  echo "No conda environment name supplied."
  exit 1
fi
CONDA_ENV_NAME=$1

if [ -z "$2" ]
then
  echo "No Python version supplied."
  exit 1
fi
PYTHON_VERSION=$2

CONDA_ENV_NAME="${CONDA_ENV_NAME}_${PYTHON_VERSION}"

if ! conda env list | grep "${CONDA_ENV_NAME}" ; then
  conda create --yes -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}"
  conda run -n "${CONDA_ENV_NAME}" --no-capture-output \
    python -m pip install --no-cache-dir --upgrade pip
  conda run -n "${CONDA_ENV_NAME}" --no-capture-output \
    python -m pip install --no-cache-dir -e ".[development,examples]"
fi
