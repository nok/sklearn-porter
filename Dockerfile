ARG PYTHON_VERSION=3.11
ARG DOTNET_VERSION=10.0

# The .NET SDK isn't available as Debian package (for all architectures), so
# it is copied from the official image (it needs the ICU library libicu72):
FROM mcr.microsoft.com/dotnet/sdk:${DOTNET_VERSION} AS dotnet

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

# Compilers and interpreters to run the transpiled estimators:
RUN apt-get update && \
    mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
        gcc libc6-dev \
        golang \
        libicu72 \
        default-jdk-headless \
        nodejs \
        php-cli \
        ruby && \
    rm -rf /var/lib/apt/lists/*

COPY --from=dotnet /usr/share/dotnet /usr/share/dotnet
ENV DOTNET_ROOT=/usr/share/dotnet \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    PATH=/usr/share/dotnet:$PATH

# GSON for the Java templates `exported`, the same version as
# `Java.GSON_DOWNLOAD_URI` (sklearn_porter/language/java/__init__.py):
ADD --chmod=644 --checksum=sha256:378534e339e6e6d50b1736fb3abb76f1c15d1be3f4c13cec6d536412e23da603 \
    https://repo1.maven.org/maven2/com/google/code/gson/gson/2.9.1/gson-2.9.1.jar \
    /opt/java/gson.jar
ENV SKLEARN_PORTER_PYTEST_GSON_PATH=/opt/java/gson.jar

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
