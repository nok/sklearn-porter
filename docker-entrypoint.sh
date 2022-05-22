#!/bin/bash
set -e

source activate sklearn-porter

exec "$@"
