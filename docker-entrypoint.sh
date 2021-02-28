#!/bin/bash
set -e

source activate base

exec "$@"

# Source:
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#entrypoint