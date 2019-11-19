#!/bin/bash
set -e

source activate ${CONDA_ENV}

exec "$@"

# Source:
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#entrypoint