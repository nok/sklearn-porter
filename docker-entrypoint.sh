#!/bin/bash
set -e

echo "gcc:    $(gcc --version | head -n 1)"
echo "dotnet: $(dotnet --version)"
echo "go:     $(go version)"
echo "java:   $(java -version 2>&1 | head -n 1)"
echo "node:   $(node --version)"
echo "php:    $(php --version | head -n 1)"
echo "ruby:   $(ruby --version)"
echo "python: $(python --version)"

exec "$@"
