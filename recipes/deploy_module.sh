#!/usr/bin/env bash

# Previous requirements:
# pip install twine

# ----------------------

source activate sklearn-porter
python setup.py sdist bdist_wheel

# Env: Test:
twine register dist/sklearn-porter-X.X.X.tar.gz -r pypitest
twine register dist/sklearn_porter-X.X.X-py2-none-any.whl -r pypitest
twine upload dist/* -r pypitest
# https://testpypi.python.org/pypi?:action=display&name=sklearn-porter

# Env: Production:
# twine register dist/sklearn-porter-X.X.X.tar.gz -r pypi
# twine register dist/sklearn_porter-X.X.X-py2-none-any.whl -r pypi
# twine upload dist/* -r pypi
# https://pypi.python.org/pypi?:action=display&name=sklearn-porter
