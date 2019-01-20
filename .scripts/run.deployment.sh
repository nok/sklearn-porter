#!/usr/bin/env bash

# Set local variables:
NAME=sklearn-porter
ANACONDA_ENV=sklearn-porter

source activate $ANACONDA_ENV

VERSION=`python -c "from sklearn_porter import __version__ as ver; print(ver);"`
COMMIT=`git rev-parse --short HEAD`

# Environment:
TARGET="https://upload.pypi.org/legacy/"
if [[ $VERSION == *"rc"* ]]; then
    TARGET="https://test.pypi.org/legacy/"
fi

# Build the package:
python ./setup.py sdist bdist_wheel

# Upload the package:
read -r -p "Upload $NAME@$VERSION (#$COMMIT) to '$TARGET'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    twine upload ./dist/* --repository-url $TARGET
fi
