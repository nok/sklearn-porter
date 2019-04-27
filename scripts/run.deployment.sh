#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd ${SCRIPT_PATH}/..

NAME=sklearn-porter

VERSION=`python -c "from sklearn_porter import __version__ as ver; print(ver);"`
COMMIT=`git rev-parse --short HEAD`

# Environment:
TARGET="pypi"
read -r -p "Do you want to use the staging environment (test.pypi.org)? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    TARGET="testpypi"
fi

# Build the package:
python ./setup.py sdist bdist_wheel

# Upload the package:
read -r -p "Upload $NAME@$VERSION (#$COMMIT) to '$TARGET'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    twine upload --repository ${TARGET} dist/*
fi
