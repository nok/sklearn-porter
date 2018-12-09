#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/..

# Dependencies:
if [[ ! -f ./gson.jar ]]; then
  wget http://central.maven.org/maven2/com/google/code/gson/gson/2.8.5/gson-2.8.5.jar
  mv gson-2.8.5.jar gson.jar
fi

# Local server:
if [[ $(python -c "import sys; print(sys.version_info[:1][0]);") == "2" ]]; then
  python -m SimpleHTTPServer 8713 &>/dev/null & serve_pid=$!;
else
  python -m http.server 8713 &>/dev/null & serve_pid=$!;
fi

# Test params:
if [[ -z "${TEST_N_RANDOM_FEATURE_SETS}" ]]; then
  TEST_N_RANDOM_FEATURE_SETS=25
fi
if [[ -z "${TEST_N_EXISTING_FEATURE_SETS}" ]]; then
  TEST_N_EXISTING_FEATURE_SETS=25
fi

# Tests:
TEST_N_RANDOM_FEATURE_SETS=${TEST_N_RANDOM_FEATURE_SETS} \
TEST_N_EXISTING_FEATURE_SETS=${TEST_N_EXISTING_FEATURE_SETS} \
    pytest tests -v -x -p no:doctest
success=$?

kill $serve_pid
rm gson.jar

exit $success
