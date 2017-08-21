#!/usr/bin/env bash

# Requirements:
# pip install twine

# Environment:
name=sklearn-porter
anaconda_env=sklearn-porter

# Set variables:
source activate $anaconda_env
rm -rf ./build/*
rm -rf ./dist/*

# Read the version:
version=`python -c "from sklearn_porter.Porter import Porter; print(Porter.__version__);"`

# Define the target environment:
target=https://test.pypi.org/legacy/
if [[ $# -eq 1 ]] ; then
    target=https://upload.pypi.org/legacy/
fi

# Build package:
python ./setup.py sdist bdist_wheel

# Upload:
read -r -p "Upload $name@$version to '$target'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    twine upload ./dist/* -r $target
fi
