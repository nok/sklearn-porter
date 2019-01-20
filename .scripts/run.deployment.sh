#!/usr/bin/env bash

# Set local variables:
NAME=sklearn-porter
ANACONDA_ENV=sklearn-porter

source activate $ANACONDA_ENV

VERSION=`python -c "from sklearn_porter import __version__ as ver; print(ver);"`
COMMIT=`git rev-parse --short HEAD`

TARGET=""

# Environment:
read -r -p "Build $NAME for production (pypi.org)? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    TARGET="https://upload.pypi.org/legacy/"
else
    TARGET="https://test.pypi.org/legacy/"
    VERSION="rc$VERSION-$COMMIT"
fi

# Build the package:
python ./setup.py sdist bdist_wheel

# Upload the package:
read -r -p "Upload $NAME@$VERSION to '$TARGET'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    twine upload ./dist/* --repository-url $TARGET
fi
