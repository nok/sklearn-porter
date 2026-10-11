.PHONY: setup lint format test test-docker docker-image notebook examples clean

IMAGE := sklearn-porter:testing
PYTHON_VERSION ?= $(shell cat .python-version)

setup:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

docker-image:
	docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE) .

test-docker: docker-image
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) \
		uv run pytest --numprocesses=auto --cov

notebook:
	uv run --group examples jupyter notebook --notebook-dir=examples/basics

examples:
	cd examples && for f in $$(find . -type f -name '*.pct.py'); do \
		uv run --group examples jupytext --to notebook --execute "$$f"; \
	done

clean:
	rm -rf build dist tmp .pytest_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
