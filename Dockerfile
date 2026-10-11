ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim

# Compilers and interpreters to run the transpiled estimators:
RUN apt-get update && \
    mkdir -p /usr/share/man/man1 && \
    apt-get install -y --no-install-recommends \
        gcc libc6-dev \
        golang \
        default-jdk-headless \
        nodejs \
        php-cli \
        ruby && \
    rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install the dependencies only; the repository is mounted at runtime:
COPY pyproject.toml uv.lock .python-version readme.md ./
RUN uv sync --locked --no-install-project

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uv", "run", "pytest"]
