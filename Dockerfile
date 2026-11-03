ARG PYTHON_VERSION=3.11
ARG DOTNET_VERSION=10.0
ARG GO_VERSION=1.27
ARG NODE_VERSION=24

# The .NET SDK isn't available as Debian package (for all architectures), so
# it is copied from the official image (it needs the ICU library libicu76):
FROM mcr.microsoft.com/dotnet/sdk:${DOTNET_VERSION} AS dotnet

# The Go and Node.js packages of Debian 13 reached their end of life, so they
# are copied from the official images:
FROM golang:${GO_VERSION}-trixie AS go
FROM node:${NODE_VERSION}-trixie-slim AS node

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-trixie-slim

# Compilers and interpreters to run the transpiled estimators:
RUN apt-get update && \
    mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
        gcc libc6-dev \
        libicu76 libstdc++6 \
        default-jdk-headless \
        php-cli \
        ruby && \
    rm -rf /var/lib/apt/lists/*

COPY --from=dotnet /usr/share/dotnet /usr/share/dotnet
COPY --from=go /usr/local/go /usr/local/go
COPY --from=node /usr/local/bin/node /usr/local/bin/node
ENV DOTNET_ROOT=/usr/share/dotnet \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    GOTOOLCHAIN=local \
    PATH=/usr/local/go/bin:/usr/share/dotnet:$PATH

# GSON for the Java templates `exported`, the same version and checksum as
# `Java.GSON_DOWNLOAD_URI` and `Java.GSON_SHA256`
# (sklearn_porter/language/java/__init__.py):
ADD --chmod=644 --checksum=sha256:96d3dd843dc2911de01cdabcd50064cb9e860353102a0a78c96fd833d396fc7d \
    https://repo1.maven.org/maven2/com/google/code/gson/gson/2.15.0/gson-2.15.0.jar \
    /opt/java/gson.jar
ENV SKLEARN_PORTER_GSON_PATH=/opt/java/gson.jar

# Use the Python of the base image instead of `.python-version`, so that
# PYTHON_VERSION selects the Python version of the tests:
ARG PYTHON_VERSION
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install the dependencies only; the repository is mounted at runtime:
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-install-project

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uv", "run", "pytest"]
