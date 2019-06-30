#!/usr/bin/env bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"
cd ${SCRIPT_PATH}/..

docker rmi -f sklearn-porter
docker build -t sklearn-porter --no-cache --force-rm .

docker rm -f $(docker ps --filter name=test -q)
docker run -d -t --name test sklearn-porter
docker exec -it test \
    make clean && conda run -n sklearn-porter \
        pytest tests -v -x -p no:doctest \
            -o python_files="*Test.py" \
            -o python_functions="test_*"

success=$?

docker rm -f $(docker ps --filter name=test -q)

exit ${success}
