#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd ${SCRIPT_PATH}/..

# Test cases:
PYTHON_FUNCTIONS="test_*"
PYTHON_FILES="*Test.py"
if [[ "$1" ]]; then
    PYTHON_FILES=${1}
fi

# Test dependencies:
if [[ ! -f ./gson.jar ]]; then
    wget http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar
    mv gson-2.8.5.jar gson.jar
fi

# Tests:
pytest tests -v -x -p no:doctest \
    -o python_files=${PYTHON_FILES} \
    -o python_functions=${PYTHON_FUNCTIONS}

exit $?
