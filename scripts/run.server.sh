#!/usr/bin/env bash

python -m http.server 8888 \
    --bind 0.0.0.0 > /dev/null 2>&1 &
