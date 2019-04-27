#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/..

# Test dependencies:
if [[ ! -f ./gson.jar ]]; then
  wget http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar
  mv gson-2.8.5.jar gson.jar
fi

# Tests:
pytest tests -v -x -p no:doctest \
    -o python_files="*Test.py" \
    -o python_functions="test_*"

exit $?
