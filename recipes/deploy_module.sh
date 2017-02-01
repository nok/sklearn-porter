#!/usr/bin/env bash

# Requirements:
# pip install twine

# Environment:
source activate sklearn-porter

# Target (e.g.: pypitest, pypi):
env=pypitest
if [[ $# -eq 1 ]] ; then
    env=$1
fi

# Version:
ver=`python -c "from sklearn_porter.Porter import Porter; print(Porter.__version__);"`

# Package:
python setup.py sdist bdist_wheel
twine register dist/sklearn-porter-$ver.tar.gz -r $env
twine register dist/sklearn_porter-$ver-py2-none-any.whl -r $env
twine upload dist/* -r $env
