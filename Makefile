.PHONY: setup lint format test test-regression test-all docker-image notebook examples clean

PYTHON_VERSION ?= $(shell cat .python-version)
IMAGE := sklearn-porter:python$(PYTHON_VERSION)

setup:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# Fast tests without the regression tests (see `addopts` in pyproject.toml):
test:
	uv run pytest --numprocesses=auto

docker-image:
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE) .

# Long running regression tests with all compilers and interpreters:
test-regression: docker-image
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) \
		uv run pytest --numprocesses=auto -m regression

# All tests with coverage:
test-all: docker-image
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) \
		uv run pytest --numprocesses=auto -m '' --cov

notebook:
	uv run --group examples jupyter notebook --notebook-dir=examples/basics

examples:
	cd examples && for f in $$(find . -type f -name '*.pct.py'); do \
		uv run --group examples jupytext --to notebook --execute "$$f"; \
	done

clean:
	rm -rf build dist tmp .pytest_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
