#!/usr/bin/env bash

# Requirements:
# pip install twine

# Environment:
name=sklearn-porter
anaconda_env=sklearn-porter

# Environment
source activate $anaconda_env
rm -rf ./build/*
rm -rf ./dist/*

# Version:
version=`python -c "from sklearn_porter.Porter import Porter; print(Porter.__version__);"`

# Target (e.g.: pypitest, pypi):
target=pypitest
if [[ $# -eq 1 ]] ; then
    target=$1
fi

# Package:
python ./setup.py sdist bdist_wheel
twine register ./dist/$name-$version.tar.gz -r $target
twine register ./dist/$name-$version-py2-none-any.whl -r $target

read -r -p "Upload $name@$version to '$target'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    twine upload ./dist/* -r $target
fi
