#!/usr/bin/env bash

# Requirements:
# pip install twine

# Environment:
name=sklearn-porter
anaconda_env=sklearn-porter
source activate $anaconda_env

# Version:
version=`python -c "from sklearn_porter.Porter import Porter; print(Porter.__version__);"`

# Target (e.g.: pypitest, pypi):
target=pypitest
if [[ $# -eq 1 ]] ; then
    target=$1
fi

# Package:
python setup.py sdist bdist_wheel
twine register dist/sklearn-porter-$version.tar.gz -r $target
twine register dist/sklearn_porter-$version-py2-none-any.whl -r $target

read -r -p "Upload $name@$version to '$target'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    twine upload dist/* -r $target
fi
