"""Test the consistency of the Docker image with the project."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_uv_version_of_docker_image():
    """The uv of the Docker image has the minor version of the build backend
    `uv_build`, because an older uv can't build the package (e.g. `uv build`
    in the container)."""
    dockerfile = (ROOT / 'Dockerfile').read_text()
    match = re.search(r'^ARG UV_VERSION=(\d+\.\d+)$', dockerfile, re.M)
    assert match, 'The Dockerfile must define ARG UV_VERSION=<major.minor>.'
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    requires = ' '.join(pyproject['build-system']['requires'])
    backend = re.search(r'uv_build>=(\d+\.\d+)\.\d+,<\d+\.\d+', requires)
    assert backend, requires
    assert match.group(1) == backend.group(1)
