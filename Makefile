SHELL := /bin/bash

export PYTHONPATH=$(shell pwd)

export CONDA_ENV_NAME=sklearn-porter
export PYTHON_VERSION=3.6

ACTIVATE_CONDA_ENV := source activate $(CONDA_ENV_NAME)_$(PYTHON_VERSION) > /dev/null 2>&1;

clean:
	resources/scripts/make_clean.sh

setup: clean
	resources/scripts/make_setup.sh ${CONDA_ENV_NAME} ${PYTHON_VERSION}

test: tests

tests:: setup
	$(ACTIVATE_CONDA_ENV) resources/scripts/make_tests.sh

test-docker: tests-docker

tests-docker:
	resources/scripts/make_tests_docker.sh

lint: setup
	$(ACTIVATE_CONDA_ENV) resources/scripts/make_lint.sh

deploy: setup
	$(ACTIVATE_CONDA_ENV) resources/scripts/make_deploy.sh

examples: setup
	$(ACTIVATE_CONDA_ENV) resources/scripts/make_examples.sh

book: notebook

notebook: setup
	$(ACTIVATE_CONDA_ENV) resources/scripts/make_notebook.sh
